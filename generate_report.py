import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, QueryOrderStatus


load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Asset Universes (match alpaca_bot.py) ─────────────────────────
GUARDIAN_ASSETS = ['SGOV', 'TLT', 'GLD']
HUNTER_ASSETS = [
    # Mega-cap Tech
    'NVDA', 'AMD', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'META',
    
    # Semiconductors
    'AVGO', 'QCOM', 'TSM', 'INTC', 'ASML',
    
    # Growth & Momentum
    'PLTR', 'NOW', 'NFLX',
    
    # Healthcare & Biotech
    'LLY', 'UNH', 'ABBV', 'MRK',
    
    # Financials
    'JPM', 'MA', 'GS',
    
    # Consumer
    'COST', 'HD',
]

# ── Timeframe configs for chart ────────────────────────────────────────
CHART_TIMEFRAMES = [
    {"key": "All", "period": "all", "timeframe": "1D"},
    {"key": "1D", "period": "1D", "timeframe": "5Min"},
]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_side(raw_side):
    """Convert 'PositionSide.LONG' → 'LONG'."""
    s = str(raw_side).upper()
    if "." in s:
        s = s.split(".")[-1]
    return s


def fetch_portfolio_history(api_key, secret_key, paper=True, period="1M", timeframe="1D"):
    base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    endpoint = f"{base_url}/v2/account/portfolio/history"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params = {
        "period": period,
        "timeframe": timeframe,
        "extended_hours": "true",
    }

    response = requests.get(endpoint, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    timestamp_key = "timestamp" if "timestamp" in data else "timestamps"
    equity_key = "equity" if "equity" in data else "equities"

    timestamps = data.get(timestamp_key, [])
    equities = data.get(equity_key, [])
    if not timestamps or not equities or len(timestamps) != len(equities):
        return pd.DataFrame(columns=["timestamp", "equity"])

    history_df = pd.DataFrame(
        [{"timestamp": ts, "equity": eq} for ts, eq in zip(timestamps, equities)]
    )
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], unit="s", utc=True)
    history_df["equity"] = pd.to_numeric(history_df["equity"], errors="coerce")
    history_df = history_df.dropna(subset=["equity"]).sort_values("timestamp").reset_index(drop=True)
    return history_df


def history_to_js_array(df):
    """Convert history DataFrame → list of [epoch_ms, equity] for JS."""
    if df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        ts_ms = int(row["timestamp"].timestamp() * 1000)
        eq = round(float(row["equity"]), 2)
        records.append([ts_ms, eq])
    return records


def fetch_benchmark_bars(api_key, secret_key, start, end):
    """Fetch daily SPY and QQQ bars from Alpaca's IEX feed, following page tokens."""
    endpoint = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params = {
        "symbols": "SPY,QQQ",
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "all",
        "feed": "iex",
        "sort": "asc",
    }
    result = {"SPY": [], "QQQ": []}

    while True:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        for symbol in result:
            result[symbol].extend(payload.get("bars", {}).get(symbol, []))

        page_token = payload.get("next_page_token")
        if not page_token:
            break
        params["page_token"] = page_token

    return result


def normalize_benchmark_series(portfolio_history_df, bars_by_symbol):
    """Scale each ETF close series to portfolio equity on its first common date."""
    normalized = {"SPY": [], "QQQ": []}
    if portfolio_history_df.empty:
        return normalized

    portfolio = portfolio_history_df[["timestamp", "equity"]].copy()
    portfolio["date"] = portfolio["timestamp"].dt.date
    portfolio = (
        portfolio.groupby("date", as_index=False).last()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        benchmark = pd.DataFrame(
            [{"timestamp": bar["t"], "close": bar["c"]} for bar in bars]
        )
        benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"], utc=True)
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.dropna(subset=["close"])
        benchmark["date"] = benchmark["timestamp"].dt.date
        benchmark = (
            benchmark.groupby("date", as_index=False).last()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        matched = portfolio.merge(benchmark[["date", "close"]], on="date", how="inner")
        if matched.empty:
            continue

        initial_equity = float(matched.iloc[0]["equity"])
        initial_close = float(matched.iloc[0]["close"])
        if initial_close <= 0:
            continue
        normalized[symbol] = [
            [
                int(row["timestamp"].timestamp() * 1000),
                round(float(row["close"]) / initial_close * initial_equity, 2),
            ]
            for _, row in matched.iterrows()
        ]

    return normalized


def calculate_risk_metrics(daily_history_df, daily_risk_free_rate=0.0):
    MIN_SHARPE_OBSERVATIONS = 20  # Need at least 20 daily returns to annualize meaningfully

    if daily_history_df.empty:
        return {"sharpe": None, "max_drawdown_pct": 0.0}

    equity = daily_history_df["equity"].astype(float)
    if len(equity) < 2:
        return {"sharpe": None, "max_drawdown_pct": 0.0}

    daily_returns = equity.pct_change().dropna()
    excess_returns = daily_returns - daily_risk_free_rate

    if len(daily_returns) < MIN_SHARPE_OBSERVATIONS or excess_returns.empty or excess_returns.std() == 0:
        sharpe_ratio = None
    else:
        sharpe_ratio = float((excess_returns.mean() / excess_returns.std()) * np.sqrt(252))

    rolling_peak = equity.cummax()
    drawdowns = (equity - rolling_peak) / rolling_peak
    max_drawdown_pct = float(drawdowns.min() * 100.0)
    return {"sharpe": sharpe_ratio, "max_drawdown_pct": max_drawdown_pct}


def build_roundtrips_from_orders(filled_orders, positions):
    """
    Pair BUY → SELL trades into roundtrips per stock (FIFO).
    For still-open positions, calculate unrealized PNL using current price.
    Returns dict: {symbol: [list of roundtrip dicts]}
    """
    from collections import defaultdict
    trades_by_symbol = defaultdict(list)
    
    # Store current position info to calculate PnL for open trades
    pos_map = {p.symbol: p for p in positions}
    
    sorted_orders = sorted(
        filled_orders,
        key=lambda order: getattr(order, "filled_at", None) or getattr(order, "submitted_at", None),
    )

    for order in sorted_orders:
        symbol = getattr(order, "symbol", "")
        if not symbol: continue
        trades_by_symbol[symbol].append(order)

    roundtrips = {}
    for symbol, trades in trades_by_symbol.items():
        open_buys = []   # Long entries awaiting a closing SELL
        open_sells = []  # Short entries awaiting a covering BUY
        rts = []

        for t in trades: # Iterate through `trades` for the current symbol
            side = _clean_side(getattr(t, "side", "")) # Use _clean_side here
            qty = _safe_float(getattr(t, "filled_qty", 0.0))
            price = _safe_float(getattr(t, "filled_avg_price", 0.0))
            dt = getattr(t, "filled_at", None) or getattr(t, "submitted_at", None)

            if side == "BUY": # Compare against "BUY"
                # First, try to match against open short lots (buy-to-cover)
                if open_sells:
                    buy_qty_remaining = qty
                    while buy_qty_remaining > 1e-9 and open_sells:
                        sell_lot = open_sells[0]
                        matched_qty = min(buy_qty_remaining, sell_lot["qty"])

                        # Short P&L: entry (sell) price - exit (buy) price
                        entry_val = matched_qty * sell_lot["price"]
                        exit_val = matched_qty * price
                        pnl = entry_val - exit_val
                        pnl_pct = (sell_lot["price"] / price - 1) * 100 if price > 0 else 0

                        order_type = str(getattr(t, "order_type", getattr(t, "type", ""))).lower()
                        if "trailing_stop" in order_type:
                            reason = "Trailing Stop"
                        else:
                            reason = "Exit Signal"

                        rts.append({
                            'entry_date': sell_lot["date"],
                            'exit_date': dt,
                            'entry_price': sell_lot["price"],
                            'exit_price': price,
                            'qty': matched_qty,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': reason,
                            'side': 'SHORT',
                            'pool': "GUARDIAN" if symbol in GUARDIAN_ASSETS else "HUNTER"
                        })

                        sell_lot["qty"] -= matched_qty
                        buy_qty_remaining -= matched_qty

                        if sell_lot["qty"] <= 1e-9:
                            open_sells.pop(0)

                    qty = buy_qty_remaining

                # Remaining qty opens a new long position
                if qty > 1e-9:
                    open_buys.append({"qty": qty, "price": price, "date": dt})

            elif side == "SELL": # Compare against "SELL"
                # First, try to match against open long lots (sell-to-close)
                if open_buys:
                    sell_qty_remaining = qty
                    
                    while sell_qty_remaining > 1e-9 and open_buys:
                        buy = open_buys[0]
                        matched_qty = min(sell_qty_remaining, buy["qty"])
                        
                        entry_val = matched_qty * buy["price"]
                        exit_val = matched_qty * price
                        pnl = exit_val - entry_val
                        pnl_pct = (price / buy["price"] - 1) * 100 if buy["price"] > 0 else 0
                        
                        # Try to infer reason based on Alpaca order type/client tag if possible
                        # Or just label it purely based on trailing stop vs. limit/market
                        order_type = str(getattr(t, "order_type", getattr(t, "type", ""))).lower()
                        if "trailing_stop" in order_type:
                            reason = "Trailing Stop"
                        else:
                            reason = "Exit Signal"

                        rts.append({
                            'entry_date': buy["date"],
                            'exit_date': dt,
                            'entry_price': buy["price"],
                            'exit_price': price,
                            'qty': matched_qty,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': reason,
                            'side': 'LONG',
                            'pool': "GUARDIAN" if symbol in GUARDIAN_ASSETS else "HUNTER"
                        })
                        
                        buy["qty"] -= matched_qty
                        sell_qty_remaining -= matched_qty
                        
                        if buy["qty"] <= 1e-9:
                            open_buys.pop(0)

                    qty = sell_qty_remaining

                # Remaining qty opens a new short position
                if qty > 1e-9:
                    open_sells.append({"qty": qty, "price": price, "date": dt})

        # Still-open positions — compute unrealized P&L
        current_price = 0.0
        if symbol in pos_map:
            current_price = _safe_float(getattr(pos_map[symbol], "current_price", 0.0))

        # Open long lots
        for buy in open_buys:
            if buy["qty"] > 1e-9:
                unrealized_pnl = (current_price - buy["price"]) * buy["qty"] if current_price else None
                unrealized_pnl_pct = (current_price / buy["price"] - 1) * 100 if current_price and buy["price"] else None
                rts.append({
                    'entry_date': buy["date"],
                    'exit_date': None,
                    'entry_price': buy["price"],
                    'exit_price': None,
                    'qty': buy["qty"],
                    'pnl': unrealized_pnl,
                    'pnl_pct': unrealized_pnl_pct,
                    'current_price': current_price,
                    'exit_reason': 'OPEN',
                    'side': 'LONG',
                    'pool': "GUARDIAN" if symbol in GUARDIAN_ASSETS else "HUNTER"
                })

        # Open short lots
        for sell_lot in open_sells:
            if sell_lot["qty"] > 1e-9:
                unrealized_pnl = (sell_lot["price"] - current_price) * sell_lot["qty"] if current_price else None
                unrealized_pnl_pct = (sell_lot["price"] / current_price - 1) * 100 if current_price and sell_lot["price"] else None
                rts.append({
                    'entry_date': sell_lot["date"],
                    'exit_date': None,
                    'entry_price': sell_lot["price"],
                    'exit_price': None,
                    'qty': sell_lot["qty"],
                    'pnl': unrealized_pnl,
                    'pnl_pct': unrealized_pnl_pct,
                    'current_price': current_price,
                    'exit_reason': 'OPEN',
                    'side': 'SHORT',
                    'pool': "GUARDIAN" if symbol in GUARDIAN_ASSETS else "HUNTER"
                })

        rts.sort(key=lambda r: r['entry_date'] if r['entry_date'] else datetime.min)
        if rts:
            roundtrips[symbol] = rts

    return roundtrips


def build_positions_dataframe(positions):
    rows = []
    for position in positions:
        rows.append(
            {
                "symbol": getattr(position, "symbol", "-"),
                "side": _clean_side(getattr(position, "side", "-")),
                "qty": _safe_float(getattr(position, "qty", 0.0)),
                "market_value": _safe_float(getattr(position, "market_value", 0.0)),
                "avg_entry_price": _safe_float(getattr(position, "avg_entry_price", 0.0)),
                "current_price": _safe_float(getattr(position, "current_price", 0.0)),
                "unrealized_pl": _safe_float(getattr(position, "unrealized_pl", 0.0)),
                "unrealized_plpc": _safe_float(getattr(position, "unrealized_plpc", 0.0)) * 100.0,
                "change_today": _safe_float(getattr(position, "change_today", 0.0)) * 100.0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "qty",
                "market_value",
                "avg_entry_price",
                "current_price",
                "unrealized_pl",
                "unrealized_plpc",
                "change_today",
            ]
        )

    return pd.DataFrame(rows)


def generate_html(account_metrics, risk_metrics, trade_stats, positions_df, equity_data_json, benchmark_data_json, heatmap_data_json, roundtrips, master_rows, paper_mode):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def usd(value):
        value = _safe_float(value)
        return ("-$" if value < 0 else "$") + f"{abs(value):,.2f}"

    def signed_usd(value):
        value = _safe_float(value)
        return ("-" if value < 0 else "+") + "$" + f"{abs(value):,.2f}"

    def js_timestamp(value):
        if value is None:
            return 0
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                return 0
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return int(timestamp.timestamp() * 1000)
        except (TypeError, ValueError, OverflowError):
            return 0

    trade_rows = []
    for symbol, symbol_roundtrips in roundtrips.items():
        for trade in symbol_roundtrips:
            if trade.get("exit_reason") == "OPEN" or trade.get("pnl") is None:
                continue
            close_ms = js_timestamp(trade.get("exit_date"))
            if not close_ms:
                continue
            entry_ms = js_timestamp(trade.get("entry_date")) or close_ms
            trade_rows.append({
                "sym": str(symbol),
                "side": str(trade.get("side", "")),
                "qty": _safe_float(trade.get("qty")),
                "entry": _safe_float(trade.get("entry_price")),
                "exit": _safe_float(trade.get("exit_price")),
                "openT": entry_ms,
                "closeT": close_ms,
                "pnl": _safe_float(trade.get("pnl")),
                "ret": _safe_float(trade.get("pnl_pct")) / 100.0,
            })
    trade_rows.sort(key=lambda trade: trade["closeT"], reverse=True)
    trades_json = json.dumps(trade_rows, separators=(",", ":"))

    symbols = sorted({trade["sym"] for trade in trade_rows})
    symbols_json = json.dumps(symbols, separators=(",", ":"))

    css = """
    :root{--bg:#eef1f6;--panel:#fff;--line:#dfe4ec;--text:#141a26;--mute:#667186;--up:#0d8a76;--down:#d4452f;--accent:#3a4fd0;color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
    @media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d121c;--panel:#151c2b;--line:#232d41;--text:#e6eaf3;--mute:#8793a9;--up:#3fc7a8;--down:#f0716b;--accent:#8ba5ff;color-scheme:dark}}
    :root[data-theme="dark"]{--bg:#0d121c;--panel:#151c2b;--line:#232d41;--text:#e6eaf3;--mute:#8793a9;--up:#3fc7a8;--down:#f0716b;--accent:#8ba5ff;color-scheme:dark}
    html{scroll-padding-top:env(safe-area-inset-top,0px)}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
    .wrap{max-width:1280px;margin:0 auto;padding:20px 16px 40px}
    header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:16px}
    h1{font-size:20px;font-weight:600;margin:0}
    h2{font-size:14px;font-weight:600;margin:0 0 12px}
    .sub{color:var(--mute);font-size:12px}
    .hr{display:flex;align-items:center;gap:10px}
    .pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:4px 10px;border:1px solid var(--line);border-radius:20px;background:var(--panel)}
    .dot{width:7px;height:7px;border-radius:50%;background:var(--up)}
    select,button{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer}
    select:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;min-width:0}
    .mt{margin-top:14px}
    .grid{display:grid;gap:14px;margin-top:14px}
    .row{grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));border-bottom:1px solid var(--line);margin:-16px -16px 16px}
    .kpis>div{padding:16px;border-right:1px solid var(--line)}
    .kpis>div:last-child{border-right:0}
    .big{font-size:26px;font-weight:600;line-height:1.2}
    .up{color:var(--up)}.down{color:var(--down)}
    .hd{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}
    .hd h2{margin:0}
    #chart{position:relative;height:320px}
    .chart-toolbar{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin:-2px 0 10px}
    .compare-buttons{display:flex;flex-wrap:wrap;gap:5px}
    .compare-btn{padding:5px 9px;font-size:11px;color:var(--mute)}
    .compare-btn.active{background:var(--accent);border-color:var(--accent);color:#fff}
    .chart-legend{display:flex;flex-wrap:wrap;align-items:center;gap:10px;font-size:11px;color:var(--mute)}
    .legend-item{display:inline-flex;align-items:center;gap:5px}
    .legend-swatch{width:12px;height:3px;border-radius:2px}
    #compareNote{min-height:16px}
    .tip{position:absolute;display:none;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font-size:12px;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.18)}
    .tip-date{display:block;margin-bottom:4px;color:var(--mute)}
    .tip-line{display:flex;align-items:center;gap:6px;margin-top:3px}
    .tip-line b{font-size:12px}.tip-line span{color:var(--text)}
    .kv{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--line)}
    .kv:last-child{border-bottom:0}.kv span{color:var(--mute)}
    .kv b .up{color:var(--up)}.kv b .down{color:var(--down)}
    .kv b small{display:block;color:var(--mute);font-size:10px;font-weight:400}
    .stats{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .split{height:8px;border-radius:4px;background:var(--down);margin-top:18px;overflow:hidden}
    .split i{display:block;height:100%;background:var(--up)}
    .scroll{overflow-x:auto}.tall{max-height:340px;overflow-y:auto}
    table{width:100%;border-collapse:collapse;font-size:13px}
    th{color:var(--mute);font-weight:500;text-align:right;white-space:nowrap;padding:8px 10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}
    td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
    th:first-child,td:first-child{text-align:left}
    tbody tr:last-child td{border-bottom:0}
    #sum tbody tr{cursor:pointer}
    #sum tbody tr:hover{background:color-mix(in srgb,var(--accent) 7%,transparent)}
    tbody tr.sel{background:color-mix(in srgb,var(--accent) 13%,transparent)}
    .tag{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
    .tag.LONG{color:var(--up);background:color-mix(in srgb,var(--up) 16%,transparent)}
    .tag.SHORT{color:var(--down);background:color-mix(in srgb,var(--down) 16%,transparent)}
    .rk{color:var(--mute);display:inline-block;width:20px}
    .bar{height:6px;min-width:90px;background:var(--line);border-radius:3px}.bar i{display:block;height:100%;border-radius:3px}
    .heat{display:grid;grid-template-rows:repeat(5,22px);gap:4px;min-width:560px}
    .heat i{border-radius:3px;background:var(--line)}
    .heat i:hover{outline:2px solid var(--text)}
    .heat b{font-size:11px;color:var(--mute);font-weight:400;align-self:center}
    .lg i{display:inline-block;width:12px;height:12px;border-radius:3px;margin:0 2px;vertical-align:-2px}
    .empty{padding:16px 10px;color:var(--mute);text-align:center}
    @media(max-width:700px){
      .wrap{padding:16px 12px 32px}
      header{align-items:flex-start}
      header>.sub{max-width:145px;text-align:right}
      .row{grid-template-columns:minmax(0,1fr)}
      #chart{height:260px}
      .hd{align-items:flex-start}
      .hd .sub{max-width:45%;text-align:right}
      .chart-toolbar{align-items:flex-start;flex-direction:column}
      .stats{gap:10px}
    }
    """

    script = r"""
    const $=s=>document.querySelector(s);
    const EQ=__EQUITY_DATA__, BENCH=__BENCHMARK_DATA__, T=__TRADE_DATA__, HEAT=__HEAT_DATA__, SYMBOLS=__SYMBOLS__;
    const fmt=n=>Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    const usd=n=>(n<0?'-':'')+'$'+fmt(Math.abs(n));
    const spl=n=>(n<0?'-':'+')+'$'+fmt(Math.abs(n));
    const pct=n=>(n<0?'':'+')+Number(n).toFixed(2)+'%';
    const cls=n=>n>=0?'up':'down';
    const pc=n=>'<span class="'+cls(n)+'">'+spl(n)+'</span>';
    const pp=n=>'<span class="'+cls(n)+'">'+pct(n)+'</span>';
    const tag=s=>'<span class="tag '+s+'">'+s+'</span>';
    const fx=n=>Number.isFinite(n)?Number(n).toFixed(2):'∞';
    const fd=t=>new Date(t).toLocaleString('en-GB',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});
    const sum=a=>a.reduce((v,t)=>v+t.pnl,0);
    const tbl=(h,r)=>'<table><thead><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+r.map(c=>'<tr>'+c.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
    const st=A=>{
      const w=A.filter(t=>t.pnl>0),l=A.filter(t=>t.pnl<=0),gp=sum(w),gl=sum(l);
      return {n:A.length,w:w.length,l:l.length,wr:A.length?w.length/A.length*100:0,gp,gl,net:gp+gl,pf:gl?gp/-gl:Infinity,best:A.length?Math.max(...A.map(t=>t.pnl)):0,worst:A.length?Math.min(...A.map(t=>t.pnl)):0};
    };
    const allStats=st(T);
    const symbolStats=SYMBOLS.map(s=>({s,...st(T.filter(t=>t.sym===s))})).sort((a,b)=>b.net-a.net);
    const seriesInfo={
      bot:{label:'Bot',color:'var(--accent)',data:EQ},
      spy:{label:'S&P 500 (SPY)',color:'#f59e0b',data:BENCH.SPY||[]},
      qqq:{label:'Nasdaq 100 (QQQ)',color:'#3b82f6',data:BENCH.QQQ||[]}
    };
    const modeSeries={bot:['bot'],spy:['bot','spy'],qqq:['bot','qqq'],all:['bot','spy','qqq']};
    let chartMode='bot';

    const symOptions=['ALL',...SYMBOLS];
    $('#pick').innerHTML=symOptions.map(s=>'<option value="'+s+'">'+(s==='ALL'?'All assets':s)+'</option>').join('');
    let selected='ALL';
    const selectedTrades=()=>selected==='ALL'?T:T.filter(t=>t.sym===selected);
    function renderPerf(){
      const periods=[['24H',1],['7D',7],['30D',30],['90D',90],['All time',Infinity]];
      const rows=periods.map(([label,days])=>{
        const cutoff=days===Infinity?-Infinity:Date.now()-days*864e5;
        const s=st(selectedTrades().filter(t=>t.closeT>=cutoff));
        return s.n?['<b>'+label+'</b>',s.n,s.wr.toFixed(0)+'%',pc(s.net),fx(s.pf),pc(s.best),pc(s.worst)]:['<b>'+label+'</b>',0,'-','-','-','-','-'];
      });
      $('#perf').innerHTML=tbl(['Period','Trades','Win Rate','Net P&amp;L','Profit Factor','Best Trade','Worst Trade'],rows);
    }
    function renderHistory(){
      const trades=selectedTrades().slice(0,selected==='ALL'?40:60);
      $('#hl').textContent=selected==='ALL'?'All assets, latest 40 trades':selected+', latest trades';
      if(!trades.length){$('#hist').innerHTML='<div class="empty">No closed trades for this selection.</div>';return;}
      $('#hist').innerHTML=tbl(['Closed','Symbol','Side','Qty','Entry','Exit','P&amp;L ($)','P&amp;L (%)','Held'],
        trades.map(t=>[fd(t.closeT),'<b>'+t.sym+'</b>',tag(t.side),fmt(t.qty),usd(t.entry),usd(t.exit),pc(t.pnl),pp(t.ret*100),((t.closeT-t.openT)/36e5).toFixed(1)+'h']));
    }
    function renderSummary(){
      const rows=symbolStats.map(x=>['<b>'+x.s+'</b>',x.n,x.wr.toFixed(0)+'%','<span class="up">'+usd(x.gp)+'</span>','<span class="down">'+usd(x.gl)+'</span>',pc(x.net),pc(x.n?x.net/x.n:0),fx(x.pf)]);
      rows.push(['<b>Total</b>',allStats.n,allStats.wr.toFixed(0)+'%','<span class="up">'+usd(allStats.gp)+'</span>','<span class="down">'+usd(allStats.gl)+'</span>',pc(allStats.net),pc(allStats.n?allStats.net/allStats.n:0),fx(allStats.pf)]);
      $('#sum').innerHTML=tbl(['Symbol','Trades','Win Rate','Gross Profit','Gross Loss','Net P&amp;L','Avg P&amp;L / Trade','Profit Factor'],rows);
      const bodyRows=[...document.querySelectorAll('#sum tbody tr')];
      bodyRows.forEach((tr,i)=>tr.onclick=()=>pick(i<symbolStats.length?symbolStats[i].s:'ALL'));
    }
    function mark(){document.querySelectorAll('#sum tbody tr').forEach((tr,i)=>tr.classList.toggle('sel',i<symbolStats.length&&symbolStats[i].s===selected));}
    function pick(s){selected=s;$('#pick').value=s;renderPerf();renderHistory();mark();}
    $('#pick').onchange=e=>pick(e.target.value);
    renderSummary();pick('ALL');

    function drawEquity(){
      const el=$('#chart'),E=EQ;
      if(!E.length){el.innerHTML='<div class="empty">No equity history was returned by Alpaca.</div>';return;}
      const requested=modeSeries[chartMode];
      const visible=requested.filter(key=>seriesInfo[key].data.length);
      const missing=requested.filter(key=>key!=='bot'&&!seriesInfo[key].data.length);
      $('#compareNote').textContent=missing.length?'Alpaca benchmark data unavailable for '+missing.map(key=>seriesInfo[key].label).join(' and ')+'.':'';
      $('#chartLegend').innerHTML=visible.map(key=>'<span class="legend-item"><i class="legend-swatch" style="background:'+seriesInfo[key].color+'"></i>'+seriesInfo[key].label+'</span>').join('');
      const W=el.clientWidth,H=el.clientHeight,m={l:62,r:12,t:10,b:24};
      const x0=E[0][0],x1=E[E.length-1][0];
      const activePoints=visible.flatMap(key=>seriesInfo[key].data);
      let lo=Math.min(...activePoints.map(p=>p[1])),hi=Math.max(...activePoints.map(p=>p[1]));
      if(hi===lo){hi+=1;lo-=1;}
      const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
      const X=t=>m.l+(t-x0)/(x1-x0||1)*(W-m.l-m.r),Y=v=>m.t+(1-(v-lo)/(hi-lo))*(H-m.t-m.b);
      const pathFor=data=>data.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('');
      const botLine=pathFor(E);
      const compareLines=visible.filter(key=>key!=='bot').map(key=>'<path d="'+pathFor(seriesInfo[key].data)+'" fill="none" stroke="'+seriesInfo[key].color+'" stroke-width="1.8" stroke-linejoin="round"/>').join('');
      const markers=visible.map(key=>'<circle id="mark-'+key+'" r="4" fill="'+seriesInfo[key].color+'" stroke="var(--panel)" stroke-width="2" style="display:none"/>').join('');
      let grid='';
      for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,y=Y(v).toFixed(1);grid+='<line x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+y+'" y2="'+y+'" stroke="var(--line)"/><text x="'+(m.l-8)+'" y="'+(+y+4)+'" text-anchor="end" fill="var(--mute)" font-size="11">'+usd(v)+'</text>';}
      for(let i=0;i<5;i++){const t=x0+(x1-x0)*i/4;grid+='<text x="'+X(t).toFixed(1)+'" y="'+(H-6)+'" text-anchor="'+(i===0?'start':i===4?'end':'middle')+'" fill="var(--mute)" font-size="11">'+new Date(t).toLocaleDateString('en-GB',{day:'numeric',month:'short'})+'</text>';}
      el.innerHTML='<svg width="'+W+'" height="'+H+'" style="display:block;touch-action:pan-y"><defs><linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--accent);stop-opacity:.25"/><stop offset="1" style="stop-color:var(--accent);stop-opacity:0"/></linearGradient></defs>'+grid+'<path d="'+botLine+'L'+X(x1)+' '+(H-m.b)+'L'+m.l+' '+(H-m.b)+'Z" fill="url(#equity-fill)"/>'+compareLines+'<path d="'+botLine+'" fill="none" stroke="var(--accent)" stroke-width="2.3" stroke-linejoin="round"/><line id="vl" y1="'+m.t+'" y2="'+(H-m.b)+'" stroke="var(--mute)" stroke-dasharray="3 3" style="display:none"/>'+markers+'</svg><div class="tip" id="tip"></div>';
      const svg=el.firstChild,tip=$('#tip'),vl=svg.querySelector('#vl');
      const closest=(data,t)=>{
        if(!data.length)return null;
        let a=0,b=data.length-1;while(a<b){const k=(a+b)>>1;data[k][0]<t?a=k+1:b=k;}
        if(a>0&&Math.abs(data[a-1][0]-t)<Math.abs(data[a][0]-t))a--;
        return data[a];
      };
      svg.onpointermove=e=>{
        const t=x0+(e.clientX-svg.getBoundingClientRect().left-m.l)/(W-m.l-m.r)*(x1-x0);
        const botPoint=closest(E,t),cx=X(botPoint[0]);
        vl.setAttribute('x1',cx);vl.setAttribute('x2',cx);vl.style.display='';
        let tooltip='<span class="tip-date">'+new Date(botPoint[0]).toLocaleString('en-GB',{weekday:'short',day:'numeric',month:'short',year:'numeric'})+'</span>';
        visible.forEach(key=>{
          const info=seriesInfo[key],point=closest(info.data,t),marker=svg.querySelector('#mark-'+key);
          if(!point)return;
          marker.setAttribute('cx',X(point[0]));marker.setAttribute('cy',Y(point[1]));marker.style.display='';
          const change=(point[1]/info.data[0][1]-1)*100;
          tooltip+='<div class="tip-line"><i class="legend-swatch" style="background:'+info.color+'"></i><b>'+info.label+'</b><span>'+usd(point[1])+'</span><strong class="'+cls(change)+'">'+pct(change)+'</strong></div>';
        });
        tip.innerHTML=tooltip;tip.style.display='block';
        tip.style.left=Math.max(0,Math.min(cx+14,W-tip.offsetWidth))+'px';tip.style.top=Math.max(0,Math.min(Y(botPoint[1])-56,H-tip.offsetHeight))+'px';
      };
      svg.onpointerleave=()=>{vl.style.display=tip.style.display='none';svg.querySelectorAll('circle').forEach(mark=>mark.style.display='none');};
    }
    drawEquity();
    window.addEventListener('resize',()=>requestAnimationFrame(drawEquity));
    if(window.ResizeObserver)new ResizeObserver(()=>requestAnimationFrame(drawEquity)).observe($('#chart'));
    document.querySelectorAll('.compare-btn').forEach(button=>button.addEventListener('click',()=>{
      chartMode=button.dataset.mode;
      document.querySelectorAll('.compare-btn').forEach(item=>{
        const active=item===button;
        item.classList.toggle('active',active);
        item.setAttribute('aria-pressed',String(active));
      });
      requestAnimationFrame(drawEquity);
    }));

    (function drawHeatmap(){
      const el=$('#heat');
      if(!HEAT.length){el.innerHTML='<div class="empty">No daily returns were returned by Alpaca.</div>';return;}
      const start=new Date(HEAT[0].date+'T00:00:00');start.setHours(0,0,0,0);start.setDate(start.getDate()-((start.getDay()+6)%7));
      const positions=HEAT.map(d=>Math.round((new Date(d.date+'T00:00:00')-start)/864e5));
      const maxAbs=Math.max(.0001,...HEAT.map(d=>Math.abs(d.change_pct)));
      const weeks=Math.floor(positions[positions.length-1]/7)+1;
      el.style.gridTemplateColumns='18px repeat('+weeks+',minmax(14px,1fr))';
      el.innerHTML=['M','T','W','T','F'].map((day,i)=>'<b style="grid-row:'+(i+1)+';grid-column:1">'+day+'</b>').join('')+
        HEAT.map((d,i)=>{
          const p=positions[i],r=Number(d.change_pct)||0,opacity=Math.round(12+88*Math.abs(r)/maxAbs);
          const color=r>=0?'--up':'--down';
          const date=new Date(d.date+'T00:00:00').toLocaleDateString('en-GB',{weekday:'short',day:'numeric',month:'short'});
          return '<i style="grid-row:'+(p%7+1)+';grid-column:'+(Math.floor(p/7)+2)+';background:color-mix(in srgb,var('+color+') '+opacity+'%,transparent)" title="'+date+': '+pct(r)+'"></i>';
        }).join('');
    })();

    $('#th').onclick=()=>{
      const root=document.documentElement;
      const dark=root.dataset.theme?root.dataset.theme==='dark':matchMedia('(prefers-color-scheme:dark)').matches;
      root.dataset.theme=dark?'light':'dark';
      requestAnimationFrame(drawEquity);
    };
    """
    script = (
        script.replace("__EQUITY_DATA__", equity_data_json)
        .replace("__BENCHMARK_DATA__", benchmark_data_json)
        .replace("__TRADE_DATA__", trades_json)
        .replace("__HEAT_DATA__", heatmap_data_json)
        .replace("__SYMBOLS__", symbols_json)
    )

    daily_class = "up" if account_metrics["daily_pl"] >= 0 else "down"
    daily_pct = (
        account_metrics["daily_pl"] / account_metrics["last_close_equity"] * 100
        if account_metrics["last_close_equity"] else 0.0
    )
    primary_metrics = [
        ("Equity balance", usd(account_metrics["equity"]),
         f'<span class="{daily_class}">{daily_pct:+.2f}%</span> vs last close'),
        ("Sharpe ratio", f'{risk_metrics["sharpe"]:.2f}' if risk_metrics["sharpe"] is not None else "—",
         "Annualised from daily returns"),
        ("Max drawdown", f'<span class="down">{risk_metrics["max_drawdown_pct"]:.2f}%</span>',
         "Peak to trough"),
        ("Daily P&amp;L", f'<span class="{daily_class}">{signed_usd(account_metrics["daily_pl"])}</span>',
         f'<span class="{daily_class}">{daily_pct:+.2f}%</span> today'),
    ]
    kpis_html = "".join(
        f'<div><div class="sub">{label}</div><div class="big">{value}</div><div class="sub">{note}</div></div>'
        for label, value, note in primary_metrics
    )

    def kv(label, value):
        return f'<div class="kv"><span>{label}</span><b>{value}</b></div>'

    snapshot_html = "".join([
        kv("Cash", usd(account_metrics["cash"])),
        kv("Last close equity", usd(account_metrics["last_close_equity"])),
        kv("Equity balance", usd(account_metrics["equity"])),
        kv("Peak equity (24H)", usd(account_metrics["peak_equity_24h"])),
        kv("All-time peak equity", usd(account_metrics["all_time_peak"])),
        kv("Open trades", account_metrics["ongoing_trades"]),
    ])

    daily_history = json.loads(heatmap_data_json)
    daily_changes = daily_history[1:] if len(daily_history) > 1 else []
    best_day = max(daily_changes, key=lambda day: day["change"], default=None)
    worst_day = min(daily_changes, key=lambda day: day["change"], default=None)

    def daily_day_value(day, color_class):
        if day is None:
            return "—"
        return (
            f'<span class="{color_class}">{signed_usd(day["change"])} '
            f'({day["change_pct"]:+.2f}%)</span>'
            f'<small>{day["date"]}</small>'
        )

    snapshot_html += "".join([
        kv("Best trading day", daily_day_value(best_day, "up")),
        kv("Worst trading day", daily_day_value(worst_day, "down")),
    ])

    total_closed = trade_stats["win_trades"] + trade_stats["loss_trades"]
    win_rate = trade_stats["win_rate_pct"]
    trade_stats_html = (
        '<div class="stats">'
        f'<div><div class="sub">Win rate</div><div class="big">{win_rate:.1f}%</div></div>'
        f'<div><div class="sub">Live trades</div><div class="big">{account_metrics["ongoing_trades"]}</div></div>'
        f'<div><div class="sub">Winning trades</div><div class="big up">{trade_stats["win_trades"]}</div></div>'
        f'<div><div class="sub">Losing trades</div><div class="big down">{trade_stats["loss_trades"]}</div></div>'
        '</div>'
        f'<div class="split"><i style="width:{win_rate:.2f}%"></i></div>'
    )

    closed_master_rows = [row for row in master_rows if row.get("closed", 0) > 0]
    ranked_rows = sorted(closed_master_rows, key=lambda item: _safe_float(item.get("total_pnl")), reverse=True)
    top_contributors = [row for row in ranked_rows if _safe_float(row.get("total_pnl")) > 0][:5]
    top_detractors = [row for row in reversed(ranked_rows) if _safe_float(row.get("total_pnl")) < 0][:5]
    visible_symbols = {row["symbol"] for row in top_contributors + top_detractors}
    visible_rows = [(index, row) for index, row in enumerate(ranked_rows, 1) if row["symbol"] in visible_symbols]
    if not visible_rows:
        visible_rows = list(enumerate(ranked_rows[:5], 1))
    max_abs_pnl = max((abs(_safe_float(row.get("total_pnl"))) for row in closed_master_rows), default=0.0) or 1.0
    attribution_rows = []
    for index, row in visible_rows:
        pnl = _safe_float(row.get("total_pnl"))
        bar_width = abs(pnl) / max_abs_pnl * 100.0
        pnl_class = "up" if pnl >= 0 else "down"
        bar_color = "--up" if pnl >= 0 else "--down"
        attribution_rows.append(
            f'<tr><td><span class="rk">{index}</span><b>{row["symbol"]}</b></td>'
            f'<td><div class="bar"><i style="width:{bar_width:.1f}%;background:var({bar_color})"></i></div></td>'
            f'<td><span class="{pnl_class}">{signed_usd(pnl)}</span></td></tr>'
        )
    if not attribution_rows:
        attribution_rows.append('<tr><td colspan="3" class="empty">No closed trades yet.</td></tr>')
    attribution_html = (
        '<div class="sub">Top 5 contributors and detractors</div>'
        '<table><thead><tr><th>Symbol</th><th>Contribution</th><th>Net P&amp;L</th></tr></thead>'
        f'<tbody>{"".join(attribution_rows)}</tbody></table>'
    )

    position_rows = []
    for _, row in positions_df.iterrows():
        side = str(row["side"]).upper()
        side_class = "LONG" if side == "LONG" else "SHORT"
        unrealized = _safe_float(row["unrealized_pl"])
        raw_change = _safe_float(row["change_today"])
        market_value = _safe_float(row["market_value"])
        daily_change = market_value * (raw_change / 100.0) / (1.0 + raw_change / 100.0) if raw_change != -100 else 0.0
        oriented_change = raw_change if side_class == "LONG" else -raw_change
        side_html = f'<span class="tag {side_class}">{side}</span>'
        unrealized_class = "up" if unrealized >= 0 else "down"
        unrealized_pct = f'<span class="{unrealized_class}">{_safe_float(row["unrealized_plpc"]):+.2f}%</span>'
        change_class = "up" if oriented_change >= 0 else "down"
        position_rows.append(
            f'<tr><td><b>{row["symbol"]}</b></td><td>{side_html}</td>'
            f'<td>{_safe_float(row["qty"]):,.4f}</td><td>{usd(market_value)}</td>'
            f'<td>{usd(row["avg_entry_price"])}</td><td>{usd(row["current_price"])}</td>'
            f'<td><span class="{unrealized_class}">{signed_usd(unrealized)}</span></td><td>{unrealized_pct}</td>'
            f'<td><span class="{change_class}">{signed_usd(daily_change)} ({oriented_change:+.2f}%)</span></td></tr>'
        )
    if not position_rows:
        position_rows.append('<tr><td colspan="9" class="empty">No open positions.</td></tr>')
    positions_html = (
        '<table><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Market Value</th>'
        '<th>Avg Price</th><th>Current Price</th><th>Unrealized P&amp;L ($)</th>'
        '<th>Unrealized P&amp;L (%)</th><th>Today’s Change</th></tr></thead>'
        f'<tbody>{"".join(position_rows)}</tbody></table>'
    )

    status = "Paper" if paper_mode else "Live"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Project Aegis Trading bot">
<title>Project Aegis Trading bot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<div class="wrap">
<header>
  <div><h1>Project Aegis Trading bot</h1><div class="sub">Alpaca account performance · Updated {now_utc}</div></div>
  <div class="hr"><span class="pill"><i class="dot"></i>{status}</span><button id="th" aria-label="Toggle light or dark theme">Theme</button></div>
</header>

<section class="panel">
  <div class="kpis" id="kpis">{kpis_html}</div>
  <h2>Equity curve</h2>
  <div class="chart-toolbar">
    <div class="compare-buttons" role="group" aria-label="Choose performance comparison">
      <button class="compare-btn active" type="button" data-mode="bot" aria-pressed="true">Bot only</button>
      <button class="compare-btn" type="button" data-mode="spy" aria-pressed="false">Bot + S&amp;P 500</button>
      <button class="compare-btn" type="button" data-mode="qqq" aria-pressed="false">Bot + Nasdaq 100</button>
      <button class="compare-btn" type="button" data-mode="all" aria-pressed="false">All 3</button>
    </div>
    <div id="chartLegend" class="chart-legend" aria-live="polite"></div>
  </div>
  <div id="compareNote" class="sub" aria-live="polite"></div>
  <div id="chart" role="img" aria-label="Alpaca portfolio equity history"></div>
</section>

<section class="grid row">
  <div class="panel"><h2>Account snapshot</h2><div id="snap">{snapshot_html}</div></div>
  <div class="panel"><h2>Trade stats</h2><div id="tstats">{trade_stats_html}</div></div>
  <div class="panel"><h2>P&amp;L attribution</h2><div id="attr" class="scroll">{attribution_html}</div></div>
</section>

<section class="panel mt"><h2>Open positions</h2><div class="scroll" id="pos">{positions_html}</div></section>

<section class="panel mt">
  <div class="hd"><h2>Asset net P&amp;L performance</h2><select id="pick" aria-label="Choose a ticker"></select></div>
  <div class="scroll" id="perf"></div>
</section>

<section class="panel mt">
  <div class="hd"><h2>Per-stock trade history</h2><span class="sub" id="hl"></span></div>
  <div class="scroll tall" id="hist"></div>
</section>

<section class="panel mt">
  <div class="hd"><h2>Overall stock summary</h2><span class="sub">Select a row to focus that ticker above</span></div>
  <div class="scroll tall" id="sum"></div>
</section>

<section class="panel mt">
  <div class="hd"><h2>Daily returns heatmap</h2>
    <span class="sub lg">Loss <i style="background:color-mix(in srgb,var(--down) 90%,transparent)"></i><i style="background:color-mix(in srgb,var(--down) 40%,transparent)"></i><i style="background:color-mix(in srgb,var(--up) 40%,transparent)"></i><i style="background:color-mix(in srgb,var(--up) 90%,transparent)"></i> Profit</span></div>
  <div class="scroll"><div class="heat" id="heat"></div></div>
</section>
</div>
<script>{script}</script>
</body>
</html>"""

def main():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    account = client.get_account()

    equity = _safe_float(getattr(account, "equity", 0.0))
    cash = _safe_float(getattr(account, "cash", 0.0))
    last_close_equity = _safe_float(getattr(account, "last_equity", 0.0), default=equity)
    daily_pl = equity - last_close_equity

    raw_histories = {}
    all_time_peak = equity  # fallback
    timeframe_by_key = {cfg["key"]: cfg for cfg in CHART_TIMEFRAMES}
    for key in ("All", "1D"):
        cfg = timeframe_by_key[key]
        try:
            hist = fetch_portfolio_history(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                paper=ALPACA_PAPER,
                period=cfg["period"],
                timeframe=cfg["timeframe"],
            )
            raw_histories[key] = hist
            if key == "All" and not hist.empty:
                all_time_peak = float(hist["equity"].max())
        except Exception as exc:
            print(f"Warning: could not fetch {key} Alpaca history: {exc}")
            raw_histories[key] = pd.DataFrame()

    history_all_df = raw_histories.get("All", pd.DataFrame())
    history_24h_df = raw_histories.get("1D", pd.DataFrame())

    risk_metrics = calculate_risk_metrics(history_all_df)
    peak_equity_24h = (
        float(history_24h_df["equity"].max()) if not history_24h_df.empty else equity
    )
    all_time_peak = max(all_time_peak, peak_equity_24h, equity)

    # ── Positions ──────────────────────────────────────────────────────
    positions = client.get_all_positions()
    positions_df = build_positions_dataframe(positions)
    ongoing_trades = len(positions)

    # ── Trade statistics ───────────────────────────────────────────────
    closed_orders = []
    until_time = None
    
    while True:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            until=until_time
        )
        batch = client.get_orders(req)
        if not batch:
            break
            
        closed_orders.extend(batch)
        
        if len(batch) < 500:
            break
            
        last_order = batch[-1]
        dt = getattr(last_order, "submitted_at", None)
        if not dt:
            break
        until_time = dt

    filled_orders = [
        order
        for order in closed_orders
        if _safe_float(getattr(order, "filled_qty", 0.0)) > 0
        and _safe_float(getattr(order, "filled_avg_price", 0.0)) > 0
    ]
    # ── Master rows & trade history calculation ───────────────────────
    roundtrips = build_roundtrips_from_orders(filled_orders, positions)
    
    master_rows = []
    for symbol, rts in roundtrips.items():
        pool = rts[0].get('pool', '—') if rts else '—'
        closed = [r for r in rts if r['exit_reason'] != 'OPEN']
        wins = [r for r in closed if r['pnl'] is not None and r['pnl'] >= 0]
        losses = [r for r in closed if r['pnl'] is not None and r['pnl'] < 0]
        stops = [r for r in closed if 'Stop' in r['exit_reason']]
        n_open = sum(1 for r in rts if r['exit_reason'] == 'OPEN')
        total_pnl = sum(r['pnl'] for r in closed if r['pnl'] is not None) if closed else 0.0
        win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
        master_rows.append({
            'symbol': symbol,
            'pool': pool,
            'total': len(rts),
            'closed': len(closed),
            'wins': len(wins),
            'losses': len(losses),
            'stops': len(stops),
            'open': n_open,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
        })

    # ── Trade statistics (derived from FIFO master_rows) ───────────────
    grand_wins = sum(r['wins'] for r in master_rows)
    grand_losses = sum(r['losses'] for r in master_rows)
    closed_total = grand_wins + grand_losses
    trade_stats = {
        "win_trades": grand_wins,
        "loss_trades": grand_losses,
        "win_rate_pct": (grand_wins / closed_total * 100.0) if closed_total else 0.0,
    }


    account_metrics = {
        "equity": equity,
        "cash": cash,
        "last_close_equity": last_close_equity,
        "daily_pl": daily_pl,
        "peak_equity_24h": peak_equity_24h,
        "all_time_peak": all_time_peak,
        "ongoing_trades": ongoing_trades,
    }

    equity_history = history_all_df if not history_all_df.empty else history_24h_df
    chart_data_json = json.dumps(history_to_js_array(equity_history), separators=(",", ":"))
    benchmark_data = {"SPY": [], "QQQ": []}
    if not equity_history.empty:
        history_start = equity_history["timestamp"].min() - pd.Timedelta(days=3)
        history_end = equity_history["timestamp"].max() + pd.Timedelta(days=2)
        try:
            benchmark_bars = fetch_benchmark_bars(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                history_start.isoformat(),
                history_end.isoformat(),
            )
            benchmark_data = normalize_benchmark_series(equity_history, benchmark_bars)
        except Exception as exc:
            print(f"Warning: could not fetch Alpaca SPY/QQQ comparison bars: {exc}")
    benchmark_data_json = json.dumps(benchmark_data, separators=(",", ":"))
    
    # ── Heatmap & Daily equity calculations ─────────────────────────────
    heatmap_data = []
    if not history_all_df.empty:
        df = history_all_df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['equity'] = df['equity'].astype(float)
        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        df = df.groupby('date', as_index=False).tail(1).sort_values('timestamp').reset_index(drop=True)
        df['daily_change'] = df['equity'].diff()
        df['daily_change_pct'] = df['equity'].pct_change() * 100
        for _, row in df.iterrows():
            chg = 0.0 if pd.isna(row['daily_change']) else float(row['daily_change'])
            chg_pct = 0.0 if pd.isna(row['daily_change_pct']) else float(row['daily_change_pct'])
            heatmap_data.append({
                "date": row['timestamp'].strftime('%Y-%m-%d'),
                "equity": float(row['equity']),
                "change": chg,
                "change_pct": chg_pct
            })
    heatmap_data_json = json.dumps(heatmap_data)

    html = generate_html(
        account_metrics, 
        risk_metrics, 
        trade_stats, 
        positions_df, 
        chart_data_json,
        benchmark_data_json,
        heatmap_data_json,
        roundtrips,
        master_rows,
        ALPACA_PAPER,
    )

    output_path = os.path.join(REPORT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
