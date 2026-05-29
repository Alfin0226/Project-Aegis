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
    {"key": "1D", "period": "1D", "timeframe": "5Min"},
    {"key": "1W", "period": "1W", "timeframe": "15Min"},
    {"key": "1M", "period": "1M", "timeframe": "1D"},
    {"key": "1Y", "period": "1A", "timeframe": "1D"},
    {"key": "All", "period": "all", "timeframe": "1D"},
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


def calculate_risk_metrics(daily_history_df):
    if daily_history_df.empty:
        return {"sharpe": 0.0, "max_drawdown_pct": 0.0}

    equity = daily_history_df["equity"].astype(float)
    if len(equity) < 2:
        return {"sharpe": 0.0, "max_drawdown_pct": 0.0}

    daily_returns = equity.pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

    rolling_peak = equity.cummax()
    drawdowns = (equity - rolling_peak) / rolling_peak
    max_drawdown_pct = float(drawdowns.min() * 100.0)
    return {"sharpe": sharpe_ratio, "max_drawdown_pct": max_drawdown_pct}


def calculate_trade_statistics(filled_orders):
    if not filled_orders:
        return {
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate_pct": 0.0,
        }

    sorted_orders = sorted(
        filled_orders,
        key=lambda order: getattr(order, "filled_at", None) or getattr(order, "submitted_at", None),
    )

    inventory = {}
    win_trades = 0
    loss_trades = 0

    for order in sorted_orders:
        symbol = getattr(order, "symbol", "")
        side = _clean_side(getattr(order, "side", "")) # Use _clean_side here
        qty = _safe_float(getattr(order, "filled_qty", 0.0))
        price = _safe_float(getattr(order, "filled_avg_price", 0.0))

        if not symbol or qty <= 0 or price <= 0:
            continue

        if symbol not in inventory:
            inventory[symbol] = {"position_qty": 0.0, "avg_price": 0.0}

        state = inventory[symbol]
        position_qty = state["position_qty"]
        avg_price = state["avg_price"]

        if side == "BUY": # Compare against "BUY"
            if position_qty < 0:
                cover_qty = min(qty, abs(position_qty))
                realized_pnl = (avg_price - price) * cover_qty
                if realized_pnl > 0:
                    win_trades += 1
                elif realized_pnl < 0:
                    loss_trades += 1
                position_qty += cover_qty
                qty -= cover_qty
                if abs(position_qty) < 1e-9:
                    position_qty = 0.0
                    avg_price = 0.0

            if qty > 0:
                if position_qty > 0:
                    total_cost = (position_qty * avg_price) + (qty * price)
                    position_qty += qty
                    avg_price = total_cost / position_qty
                else:
                    position_qty = qty
                    avg_price = price

        elif side == "SELL": # Compare against "SELL"
            if position_qty > 0:
                close_qty = min(qty, position_qty)
                realized_pnl = (price - avg_price) * close_qty
                if realized_pnl > 0:
                    win_trades += 1
                elif realized_pnl < 0:
                    loss_trades += 1
                position_qty -= close_qty
                qty -= close_qty
                if abs(position_qty) < 1e-9:
                    position_qty = 0.0
                    avg_price = 0.0

            if qty > 0:
                if position_qty < 0:
                    total_proceeds_basis = (abs(position_qty) * avg_price) + (qty * price)
                    position_qty -= qty
                    avg_price = total_proceeds_basis / abs(position_qty)
                else:
                    position_qty = -qty
                    avg_price = price

        state["position_qty"] = position_qty
        state["avg_price"] = avg_price

    closed_trades = win_trades + loss_trades
    win_rate_pct = (win_trades / closed_trades * 100.0) if closed_trades > 0 else 0.0
    return {
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate_pct": win_rate_pct,
    }

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
        open_buys = []
        rts = []

        for t in trades: # Iterate through `trades` for the current symbol
            side = _clean_side(getattr(t, "side", "")) # Use _clean_side here
            qty = _safe_float(getattr(t, "filled_qty", 0.0))
            price = _safe_float(getattr(t, "filled_avg_price", 0.0))
            dt = getattr(t, "filled_at", None) or getattr(t, "submitted_at", None)

            if side == "BUY": # Compare against "BUY"
                # We simply track the qty left in this buy order
                open_buys.append({"qty": qty, "price": price, "date": dt})
            elif side == "SELL" and open_buys: # Compare against "SELL"
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
                        'pool': "GUARDIAN" if symbol in GUARDIAN_ASSETS else "HUNTER"
                    })
                    
                    buy["qty"] -= matched_qty
                    sell_qty_remaining -= matched_qty
                    
                    if buy["qty"] <= 1e-9:
                        open_buys.pop(0)

        # Still-open positions
        current_price = 0.0
        if symbol in pos_map:
            current_price = _safe_float(getattr(pos_map[symbol], "current_price", 0.0))

        for buy in open_buys:
            if buy["qty"] > 1e-9:
                rts.append({
                    'entry_date': buy["date"],
                    'exit_date': None,
                    'entry_price': buy["price"],
                    'exit_price': None,
                    'qty': buy["qty"],
                    'pnl': None,
                    'pnl_pct': None,
                    'current_price': current_price,
                    'exit_reason': 'OPEN',
                    'pool': "GUARDIAN" if symbol in GUARDIAN_ASSETS else "HUNTER"
                })

        rts.sort(key=lambda r: r['entry_date'] if r['entry_date'] else datetime.min)
        if rts:
            roundtrips[symbol] = rts

    return roundtrips

def generate_trade_history_html(roundtrips, master_rows):
    """Generate accordion-based HTML for each stock's roundtrip trades + overall summary."""
    css = """
    <style>
        .th-section { margin-top: 32px; }
        .th-header { font-size: 18px; font-weight: 600; margin-bottom: 16px; border-bottom: 2px solid #1e2d44; padding-bottom: 8px; color: #f8fafc; }
        
        details.stock-accordion {
            margin-bottom: 10px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #1d2d44;
            background: #0e1622;
        }
        details.stock-accordion summary::-webkit-details-marker {
            display: none;
        }
        details.stock-accordion summary {
            list-style: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 18px;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s;
        }
        details.stock-accordion summary:hover {
            background: #162235;
            border-color: #3b82f6;
        }
        details.stock-accordion[open] summary {
            background: #162235;
            border-bottom: 1px solid #1d2d44;
        }
        .accordion-content {
            background: #090e17;
            padding: 20px;
        }
        .acc-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .acc-symbol {
            font-size: 18px;
            font-weight: 700;
            color: #f8fafc;
        }
        .acc-badge {
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(59, 130, 246, 0.12);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.25);
        }
        .acc-right {
            display: flex;
            align-items: center;
            gap: 32px;
        }
        .acc-pnl-wrap, .acc-wr-wrap, .acc-open-wrap {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        .acc-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 3px;
        }
        .acc-val {
            font-size: 14px;
            font-weight: 600;
        }
        .chevron {
            color: #64748b;
            transition: transform 0.2s;
            font-size: 11px;
        }
        details.stock-accordion[open] .chevron {
            transform: rotate(90deg);
        }
        
        .th-table-wrap { background: #0e1622; border: 1px solid #1d2d44; border-radius: 6px; overflow: hidden; margin-bottom: 16px; }
        .th-table { width: 100%; border-collapse: collapse; }
        .th-table th { background: #0a0f18; color: #cbd5e1; font-weight: 600; padding: 10px; text-align: left; font-size: 12px; white-space: nowrap; border-bottom: 1px solid #1d2d44; }
        .th-table td { padding: 9px 10px; border-bottom: 1px solid #1d2d44; font-size: 13px; white-space: nowrap; color: #e2e8f0; }
        .th-table tbody tr:hover { background: #162235; }
        .pnl-pos { color: #10b981; font-weight: 600; }
        .pnl-neg { color: #f43f5e; font-weight: 600; }
        .pnl-open { color: #94a3b8; font-style: italic; }
        .reason-badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; display: inline-block; }
        .reason-stop { background: rgba(244,63,94,0.12); color: #fca5a5; border: 1px solid rgba(244,63,94,0.25); }
        .reason-exit { background: rgba(245,158,11,0.12); color: #fcd34d; border: 1px solid rgba(245,158,11,0.25); }
        .reason-open { background: rgba(16,185,129,0.12); color: #86efac; border: 1px solid rgba(16,185,129,0.25); }
        .summary-wrap { background: #0a0f18; padding: 12px; border-radius: 6px; border: 1px solid #1d2d44; }
        .summary-table { width: 100%; border-collapse: collapse; }
        .summary-table th { color: #64748b; font-size: 11px; text-align: center; padding: 4px; border: none; background: transparent; text-transform: uppercase; letter-spacing: 0.05em; }
        .summary-table td { text-align: center; font-size: 13px; font-weight: 600; padding: 4px; border: none; }
        .master-summary { margin-top: 40px; }
        .master-table th { background: #0a0f18; color: #f8fafc; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
        .total-row td { background: #0e1622; font-weight: bold; border-top: 2px solid #1d2d44; color: #f8fafc; }
    </style>
    """

    def fmt_date(d):
        if not d: return '—'
        if isinstance(d, datetime):
            return d.strftime('%Y-%m-%d')
        return str(d)[:10]

    def fmt_price(p):
        return f'${p:,.2f}' if p is not None else '—'

    def fmt_pnl(pnl, pnl_pct):
        if pnl is None:
            return '<span class="pnl-open">OPEN</span>'
        cls = 'pnl-pos' if pnl >= 0 else 'pnl-neg'
        sign = '+' if pnl >= 0 else ''
        return f'<span class="{cls}">{sign}${pnl:,.2f} ({sign}{pnl_pct:.1f}%)</span>'

    def fmt_reason(reason):
        tag_map = {
            'Trailing Stop': ('📉 Trailing Stop', 'reason-stop'),
            'Exit Signal': ('🔴 Exit Signal', 'reason-exit'),
            'OPEN': ('🟢 Open', 'reason-open'),
        }
        label, cls = tag_map.get(reason, (reason, 'reason-exit'))
        return f'<span class="reason-badge {cls}">{label}</span>'

    html_parts = [css, '<div class="th-section">']
    html_parts.append('<div class="th-header">Per-Stock Trade History</div>')

    row_map = {r['symbol']: r for r in master_rows}

    for symbol in sorted(roundtrips.keys()):
        rts = roundtrips[symbol]
        row_data = row_map.get(symbol, {})
        
        pool = rts[0].get('pool', '—') if rts else '—'
        total_pnl = row_data.get('total_pnl', 0.0)
        win_rate = row_data.get('win_rate', 0.0)
        n_closed = row_data.get('closed', 0)
        n_open = row_data.get('open', 0)
        
        pnl_cls = 'pnl-pos' if total_pnl >= 0 else 'pnl-neg'
        pnl_sign = '+' if total_pnl >= 0 else ''
        pnl_str = f"{pnl_sign}${total_pnl:,.2f}"
        
        wr_cls = 'pnl-pos' if win_rate >= 50 else ('pnl-neg' if n_closed > 0 else '')
        wr_str = f"{win_rate:.0f}%" if n_closed > 0 else '—'

        html_parts.append(f"""
        <details class="stock-accordion">
            <summary>
                <div class="acc-left">
                    <span class="chevron">▶</span>
                    <span class="acc-symbol">{symbol}</span>
                    <span class="acc-badge">{pool}</span>
                </div>
                <div class="acc-right">
                    <div class="acc-pnl-wrap">
                        <span class="acc-label">Net P&amp;L</span>
                        <span class="acc-val {pnl_cls}">{pnl_str}</span>
                    </div>
                    <div class="acc-wr-wrap">
                        <span class="acc-label">Win Rate</span>
                        <span class="acc-val {wr_cls}">{wr_str}</span>
                    </div>
                    <div class="acc-open-wrap">
                        <span class="acc-label">Trades</span>
                        <span class="acc-val">{n_closed} closed {f'/ {n_open} open' if n_open > 0 else ''}</span>
                    </div>
                </div>
            </summary>
            <div class="accordion-content">
        """)

        html_parts.append('<div class="th-table-wrap"><table class="th-table">')
        html_parts.append(
            '<thead><tr><th>#</th><th>Entry Date</th><th>Exit Date</th>'
            '<th>Qty</th><th>Entry Price</th><th>Exit Price</th>'
            '<th>P&amp;L</th><th>Exit Reason</th></tr></thead><tbody>'
        )

        closed = [r for r in rts if r['pnl'] is not None]
        wins = [r for r in closed if r['pnl'] >= 0]
        losses = [r for r in closed if r['pnl'] < 0]
        stops = [r for r in closed if 'Stop' in r['exit_reason']]

        for i, rt in enumerate(rts, 1):
            qty_str = f'{rt["qty"]:,.2f}' if rt['qty'] else '—'
            html_parts.append(
                f'<tr>'
                f'<td>{i}</td>'
                f'<td>{fmt_date(rt["entry_date"])}</td>'
                f'<td>{fmt_date(rt["exit_date"])}</td>'
                f'<td>{qty_str}</td>'
                f'<td>{fmt_price(rt["entry_price"])}</td>'
                f'<td>{fmt_price(rt["exit_price"])}</td>'
                f'<td>{fmt_pnl(rt["pnl"], rt["pnl_pct"])}</td>'
                f'<td>{fmt_reason(rt["exit_reason"])}</td>'
                f'</tr>'
            )

        html_parts.append('</tbody></table></div>')

        avg_win = sum(r['pnl'] for r in wins)/len(wins) if wins else 0
        avg_loss = sum(r['pnl'] for r in losses)/len(losses) if losses else 0

        html_parts.append('<div class="summary-wrap"><table class="summary-table">')
        html_parts.append(
            '<thead><tr><th>Closed</th><th>Wins</th><th>Losses</th>'
            '<th>Stops</th><th>Open</th><th>Win Rate</th>'
            '<th>Avg Win</th><th>Avg Loss</th><th>Net P&amp;L</th></tr></thead><tbody>'
        )
        pnl_cls = 'pnl-pos' if total_pnl >= 0 else 'pnl-neg'
        pnl_sign = '+' if total_pnl >= 0 else ''
        html_parts.append(
            f'<tr>'
            f'<td>{len(closed)}</td>'
            f'<td class="pnl-pos">{len(wins)}</td>'
            f'<td class="pnl-neg">{len(losses)}</td>'
            f'<td>{len(stops)}</td>'
            f'<td>{n_open}</td>'
            f'<td class="{wr_cls}">{win_rate:.0f}%</td>'
            f'<td class="pnl-pos">{fmt_price(avg_win)}</td>'
            f'<td class="pnl-neg">{fmt_price(avg_loss)}</td>'
            f'<td class="{pnl_cls}">{pnl_sign}${total_pnl:,.2f}</td>'
            f'</tr>'
        )
        html_parts.append('</tbody></table></div></div></details>')

    # --- Master Summary ---
    html_parts.append('<div class="master-summary">')
    html_parts.append('<div class="th-header">Overall Stock Summary Table</div>')
    html_parts.append('<div class="th-table-wrap"><table class="th-table master-table">')
    html_parts.append(
        '<thead><tr><th>Stock</th><th>Pool</th><th>Total</th><th>Closed</th>'
        '<th>Wins</th><th>Losses</th><th>Stops</th><th>Open</th>'
        '<th>Win Rate</th><th>Net P&amp;L</th></tr></thead><tbody>'
    )

    grand_total = grand_closed = grand_wins = grand_losses = grand_stops = grand_open = 0
    grand_pnl = 0.0

    for row in sorted(master_rows, key=lambda r: r['total_pnl'], reverse=True):
        wr_cls = 'pnl-pos' if row['win_rate'] >= 50 else ('pnl-neg' if row['closed'] > 0 else '')
        pnl_cls = 'pnl-pos' if row['total_pnl'] >= 0 else 'pnl-neg'
        pnl_sign = '+' if row['total_pnl'] >= 0 else ''
        
        html_parts.append(
            f'<tr>'
            f'<td><strong>{row["symbol"]}</strong></td>'
            f'<td>{row["pool"]}</td>'
            f'<td>{row["total"]}</td>'
            f'<td>{row["closed"]}</td>'
            f'<td class="pnl-pos">{row["wins"]}</td>'
            f'<td class="pnl-neg">{row["losses"]}</td>'
            f'<td>{row["stops"]}</td>'
            f'<td>{row["open"]}</td>'
            f'<td class="{wr_cls}">{row["win_rate"]:.0f}%</td>'
            f'<td class="{pnl_cls}">{pnl_sign}${row["total_pnl"]:,.2f}</td>'
            f'</tr>'
        )
        grand_total += row['total']
        grand_closed += row['closed']
        grand_wins += row['wins']
        grand_losses += row['losses']
        grand_stops += row['stops']
        grand_open += row['open']
        grand_pnl += row['total_pnl']

    grand_wr = (grand_wins / grand_closed * 100) if grand_closed > 0 else 0
    wr_cls = 'pnl-pos' if grand_wr >= 50 else 'pnl-neg'
    pnl_cls = 'pnl-pos' if grand_pnl >= 0 else 'pnl-neg'
    pnl_sign = '+' if grand_pnl >= 0 else ''
    
    html_parts.append(
        f'<tr class="total-row">'
        f'<td>TOTAL</td><td>&mdash;</td>'
        f'<td>{grand_total}</td>'
        f'<td>{grand_closed}</td>'
        f'<td class="pnl-pos">{grand_wins}</td>'
        f'<td class="pnl-neg">{grand_losses}</td>'
        f'<td>{grand_stops}</td>'
        f'<td>{grand_open}</td>'
        f'<td class="{wr_cls}">{grand_wr:.0f}%</td>'
        f'<td class="{pnl_cls}">{pnl_sign}${grand_pnl:,.2f}</td>'
        f'</tr>'
    )
    
    html_parts.append('</tbody></table></div></div></div>')
    return '\n'.join(html_parts)


def generate_equity_table_html(daily_history_df):
    """Generate the daily performance heatmap container placeholder."""
    html = '''
    <div class="panel heatmap-panel" style="margin-top: 32px;">
        <div class="th-header">Daily Returns Heatmap</div>
        <div class="heatmap-wrapper">
            <div id="heatmapGrid" class="heatmap-grid"></div>
        </div>
        <div class="heatmap-legend">
            <span class="legend-text">Significant Loss</span>
            <div class="legend-scale">
                <div class="legend-cell" style="background: rgba(244, 63, 94, 0.95)"></div>
                <div class="legend-cell" style="background: rgba(244, 63, 94, 0.65)"></div>
                <div class="legend-cell" style="background: rgba(244, 63, 94, 0.35)"></div>
                <div class="legend-cell" style="background: #1e293b"></div>
                <div class="legend-cell" style="background: rgba(16, 185, 129, 0.35)"></div>
                <div class="legend-cell" style="background: rgba(16, 185, 129, 0.65)"></div>
                <div class="legend-cell" style="background: rgba(16, 185, 129, 0.95)"></div>
            </div>
            <span class="legend-text">Significant Profit</span>
        </div>
    </div>
    '''
    return html


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


def fmt_money(value):
    return f"${value:,.2f}"


def fmt_pct(value):
    return f"{value:+,.2f}%"


def generate_html(account_metrics, risk_metrics, trade_stats, positions_df, chart_data_json, trade_history_html, equity_table_html, heatmap_data_json, master_rows):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    daily_pl_class = "green" if account_metrics["daily_pl"] >= 0 else "red"
    mdd_class = "green" if risk_metrics["max_drawdown_pct"] >= -5 else "red"

    total_closed = trade_stats["win_trades"] + trade_stats["loss_trades"]
    if total_closed == 0 and account_metrics["ongoing_trades"] > 0:
        trade_note = (
            '<p class="note">All current positions are still open &mdash; '
            "trade stats update once a position is fully closed (buy &rarr; sell round-trip).</p>"
        )
    elif total_closed == 0:
        trade_note = '<p class="note">No trades have been executed yet.</p>'
    else:
        trade_note = ""

    # Position rows with unrealized P&L double-sided progress bar
    position_rows = ""
    if positions_df.empty:
        position_rows = (
            "<tr><td colspan='9' style='text-align:center;color:#94a3b8;'>"
            "No Open Positions</td></tr>"
        )
    else:
        for _, row in positions_df.iterrows():
            pl_class = "green" if row["unrealized_pl"] >= 0 else "red"
            change_class = "green" if row["change_today"] >= 0 else "red"
            
            pnl_pc = row['unrealized_plpc']
            clamped = max(-10.0, min(10.0, pnl_pc))
            if clamped >= 0:
                bar_class = "pos"
                bar_width = (clamped / 10.0) * 50
                bar_margin = 50
            else:
                bar_class = "neg"
                bar_width = (abs(clamped) / 10.0) * 50
                bar_margin = 50 - bar_width
                
            position_rows += f"""
            <tr>
                <td><strong>{row['symbol']}</strong></td>
                <td><span class="side-badge">{row['side']}</span></td>
                <td>{row['qty']:,.4f}</td>
                <td>{fmt_money(row['market_value'])}</td>
                <td>{fmt_money(row['avg_entry_price'])}</td>
                <td>{fmt_money(row['current_price'])}</td>
                <td class="{pl_class}">{fmt_money(row['unrealized_pl'])}</td>
                <td class="{pl_class}">
                    <div class="pnl-progress-flex">
                        <span class="pnl-pc-text">{fmt_pct(row['unrealized_plpc'])}</span>
                        <div class="pnl-progress-track">
                            <div class="pnl-progress-bar {bar_class}" style="width: {bar_width:.1f}%; margin-left: {bar_margin:.1f}%;"></div>
                            <div class="pnl-progress-center"></div>
                        </div>
                    </div>
                </td>
                <td class="{change_class}">{fmt_pct(row['change_today'])}</td>
            </tr>
            """

    # --- P&L Attribution Breakdown HTML ---
    pos_stocks = [r for r in master_rows if r['total_pnl'] > 0]
    total_gains = sum(r['total_pnl'] for r in pos_stocks)
    
    attribution_html = ""
    if total_gains == 0:
        attribution_html = '<p class="note">No positive closed P&L to attribute yet.</p>'
    else:
        sorted_pos = sorted(pos_stocks, key=lambda x: x['total_pnl'], reverse=True)
        attribution_html += '<div class="attribution-list">'
        for r in sorted_pos[:6]:
            pct = (r['total_pnl'] / total_gains) * 100.0
            attribution_html += f"""
            <div class="attrib-row">
                <div class="attrib-meta">
                    <span class="attrib-sym"><strong>{r['symbol']}</strong></span>
                    <span class="attrib-val green">+{fmt_money(r['total_pnl'])}</span>
                </div>
                <div class="attrib-progress-container">
                    <div class="attrib-progress-bar" style="width: {pct:.1f}%"></div>
                    <span class="attrib-pct">{pct:.1f}%</span>
                </div>
            </div>
            """
        if len(sorted_pos) > 6:
            other_gains = sum(r['total_pnl'] for r in sorted_pos[6:])
            other_pct = (other_gains / total_gains) * 100.0
            attribution_html += f"""
            <div class="attrib-row">
                <div class="attrib-meta">
                    <span class="attrib-sym"><strong>Others</strong></span>
                    <span class="attrib-val green">+{fmt_money(other_gains)}</span>
                </div>
                <div class="attrib-progress-container">
                    <div class="attrib-progress-bar" style="width: {other_pct:.1f}%; background: #475569;"></div>
                    <span class="attrib-pct">{other_pct:.1f}%</span>
                </div>
            </div>
            """
        attribution_html += '</div>'

    # --- Horizontal Stock Summary Bar Chart ---
    sorted_master = sorted(master_rows, key=lambda x: x['total_pnl'], reverse=True)
    max_abs_pnl = max(abs(r['total_pnl']) for r in master_rows) if master_rows else 1.0
    if max_abs_pnl == 0.0:
        max_abs_pnl = 1.0
        
    bar_chart_html = """
    <div class="panel bar-chart-panel">
        <h3>Asset Net P&amp;L Performance</h3>
        <div class="bar-chart-container">
    """
    for r in sorted_master:
        pnl = r['total_pnl']
        pnl_pct_of_max = (abs(pnl) / max_abs_pnl) * 50.0
        
        pnl_val_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        pnl_class = "green" if pnl >= 0 else "red"
        
        if pnl >= 0:
            bar_style = f"width: {pnl_pct_of_max:.1f}%; margin-left: 50%;"
            bar_color_class = "bar-pos"
        else:
            bar_style = f"width: {pnl_pct_of_max:.1f}%; margin-left: {50.0 - pnl_pct_of_max:.1f}%;"
            bar_color_class = "bar-neg"
            
        bar_chart_html += f"""
        <div class="bar-chart-row">
            <div class="bar-chart-label"><strong>{r['symbol']}</strong> <span class="bar-chart-pool">{r['pool']}</span></div>
            <div class="bar-chart-track-wrap">
                <div class="bar-chart-track">
                    <div class="bar-chart-fill {bar_color_class}" style="{bar_style}"></div>
                    <div class="bar-chart-grid-zero"></div>
                </div>
            </div>
            <div class="bar-chart-value {pnl_class}">{pnl_val_str}</div>
        </div>
        """
    bar_chart_html += """
        </div>
    </div>
    """

    win_rate_val = trade_stats['win_rate_pct']
    win_dash = f"{win_rate_val:.2f}"
    loss_dash = f"{100.0 - win_rate_val:.2f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-4MRR5QJQXN"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', 'G-4MRR5QJQXN');
    </script>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Trading Bot Dashboard</title>
    <style>
        *{{ box-sizing:border-box; }}
        body{{ margin:0; background:#060b13; color:#f8fafc; font-family:Inter,Segoe UI,Arial,sans-serif; -webkit-font-smoothing: antialiased; }}
        .container{{ max-width:1280px; margin:0 auto; padding:24px; }}
        .header{{ margin-bottom:24px; display:flex; justify-content:space-between; align-items:flex-end; border-bottom: 1px solid #1d2d44; padding-bottom: 16px; }}
        .title{{ margin:0; font-size:26px; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; }}
        .subtitle{{ margin:4px 0 0; color:#64748b; font-size:12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }}
        
        /* ── KPI Cards ── */
        .cards{{ display:grid; grid-template-columns:repeat(4,minmax(180px,1fr)); gap:16px; margin-bottom:24px; }}
        .card{{ background:#0e1622; border:1px solid #1d2d44; border-radius:8px; padding:18px 20px; display:flex; flex-direction:column; justify-content:space-between; transition:all 0.2s; }}
        .card:hover {{ border-color: #3b82f6; transform: translateY(-2px); }}
        .card-label{{ color:#64748b; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.07em; margin-bottom: 8px; }}
        .card-value{{ font-size:32px; font-weight:800; color:#f8fafc; line-height:1.1; }}
        
        .green{{ color:#10b981 !important; }} 
        .red{{ color:#f43f5e !important; }} 
        .text-blue{{ color:#3b82f6 !important; }}
        
        /* ── Sections Grid ── */
        .sections{{ display:grid; grid-template-columns:1fr 1.1fr 0.9fr; gap:16px; margin-bottom:24px; }}
        .panel{{ background:#0e1622; border:1px solid #1d2d44; border-radius:8px; padding:18px; }}
        .panel h3{{ margin:0 0 16px; font-size:15px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color: #94a3b8; border-bottom: 1px solid #1d2d44; padding-bottom: 8px; }}
        
        /* ── Snapshot Panel ── */
        .stats-grid{{ display:grid; grid-template-columns:repeat(2,minmax(100px,1fr)); gap:12px; }}
        .stat-box{{ background:#090e17; border:1px solid #1d2d44; border-radius:6px; padding:12px 14px; }}
        .stat-box.full-width {{ grid-column: span 2; }}
        .stat-box .label {{ font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
        .stat-box .value {{ font-size: 18px; font-weight: 700; color: #f8fafc; }}
        
        /* ── Trade Stats Donut ── */
        .trade-stats-container {{ display: flex; align-items: center; gap: 20px; }}
        .donut-chart-wrap {{ width: 120px; height: 120px; flex-shrink: 0; position: relative; }}
        .donut-svg {{ transform: rotate(-90deg); width: 100%; height: 100%; }}
        .donut-center-pct {{ fill: #f8fafc; font-size: 6px; font-weight: 800; text-anchor: middle; transform: rotate(90deg); transform-origin: center; }}
        .donut-center-sub {{ fill: #64748b; font-size: 2px; font-weight: 600; text-anchor: middle; transform: rotate(90deg); transform-origin: center; text-transform: uppercase; letter-spacing: 0.05em; }}
        .trade-stats-details {{ flex-grow: 1; display: flex; flex-direction: column; gap: 8px; }}
        .stat-row {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #162235; padding-bottom: 4px; }}
        .stat-row:last-child {{ border-bottom: none; }}
        .stat-row .stat-label {{ font-size: 12px; color: #94a3b8; }}
        .stat-row .stat-val {{ font-size: 13px; font-weight: 700; }}
        
        /* ── P&L Attribution Breakdown ── */
        .attribution-list {{ display: flex; flex-direction: column; gap: 10px; }}
        .attrib-row {{ display: flex; flex-direction: column; gap: 4px; }}
        .attrib-meta {{ display: flex; justify-content: space-between; font-size: 12px; }}
        .attrib-sym {{ color: #e2e8f0; }}
        .attrib-val {{ font-weight: 600; }}
        .attrib-progress-container {{ display: flex; align-items: center; gap: 8px; }}
        .attrib-progress-bar {{ height: 5px; background: #10b981; border-radius: 3px; }}
        .attrib-pct {{ font-size: 11px; color: #64748b; font-weight: 600; flex-shrink: 0; width: 32px; text-align: right; }}
        
        /* ── Position Table Double-Sided P&L Progress Bar ── */
        .pnl-progress-flex {{ display: flex; align-items: center; gap: 12px; width: 100%; }}
        .pnl-pc-text {{ font-size: 13px; font-weight: 700; width: 55px; text-align: right; flex-shrink: 0; }}
        .pnl-progress-track {{ height: 6px; background: #1a2436; border-radius: 3px; flex-grow: 1; position: relative; overflow: hidden; display: flex; align-items: center; border: 1px solid #1d2d44; }}
        .pnl-progress-bar {{ height: 100%; border-radius: 2px; }}
        .pnl-progress-bar.pos {{ background: #10b981; }}
        .pnl-progress-bar.neg {{ background: #f43f5e; }}
        .pnl-progress-center {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #64748b; opacity: 0.8; }}
        .side-badge {{ padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }}
        
        /* ── Horizontal Bar Chart ── */
        .bar-chart-panel {{ background:#0e1622; border:1px solid #1d2d44; border-radius:8px; padding:18px; }}
        .bar-chart-container {{ display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }}
        .bar-chart-row {{ display: flex; align-items: center; gap: 16px; height: 26px; }}
        .bar-chart-label {{ font-size: 12px; color: #f8fafc; width: 90px; flex-shrink: 0; text-align: left; display: flex; align-items: center; gap: 6px; }}
        .bar-chart-pool {{ font-size: 8px; color: #64748b; background: #1e293b; padding: 1px 4px; border-radius: 3px; font-weight: normal; }}
        .bar-chart-track-wrap {{ flex-grow: 1; height: 100%; display: flex; align-items: center; }}
        .bar-chart-track {{ height: 14px; background: #0a0f18; border: 1px solid #1d2d44; border-radius: 3px; flex-grow: 1; position: relative; overflow: hidden; }}
        .bar-chart-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s ease; }}
        .bar-chart-fill.bar-pos {{ background: linear-gradient(90deg, rgba(16, 185, 129, 0.6), rgba(16, 185, 129, 0.95)); }}
        .bar-chart-fill.bar-neg {{ background: linear-gradient(90deg, rgba(244, 63, 94, 0.95), rgba(244, 63, 94, 0.6)); }}
        .bar-chart-grid-zero {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #64748b; z-index: 2; }}
        .bar-chart-value {{ font-size: 12px; font-weight: 700; width: 85px; text-align: right; flex-shrink: 0; }}
        
        /* ── Calendar Performance Heatmap ── */
        .heatmap-panel {{ background:#0e1622; border:1px solid #1d2d44; border-radius:8px; padding:18px; }}
        .heatmap-wrapper {{ position: relative; margin-top: 24px; overflow-x: auto; }}
        .heatmap-grid {{ display: inline-block; min-width: 100%; padding-top: 20px; }}
        .heatmap-grid-inner {{ display: flex; gap: 6px; }}
        .heatmap-cols-wrap {{ display: flex; gap: 3px; position: relative; }}
        .heatmap-week-col {{ display: flex; flex-direction: column; gap: 3px; }}
        .heatmap-cell {{ width: 11px; height: 11px; background: #090e17; border: 1px solid #162235; border-radius: 2px; cursor: pointer; position: relative; }}
        .heatmap-cell:hover {{ border-color: #3b82f6 !important; transform: scale(1.15); z-index: 5; }}
        .heatmap-cell.empty-cell {{ background: transparent; border: none; cursor: default; pointer-events: none; }}
        .heatmap-day-labels {{ display: flex; flex-direction: column; gap: 3px; font-size: 9px; color: #64748b; justify-content: space-between; padding-right: 6px; height: 95px; width: 28px; text-align: left; text-transform: uppercase; font-weight: 600; }}
        .heatmap-months-labels {{ position: absolute; top: -18px; left: 34px; right: 0; height: 16px; font-size: 10px; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
        .heatmap-legend {{ display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 16px; font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .legend-scale {{ display: flex; gap: 2px; }}
        .legend-cell {{ width: 10px; height: 10px; border-radius: 1px; }}
        .heatmap-tooltip {{ position: absolute; background: #0a0f18; border: 1px solid #3b82f6; border-radius: 4px; padding: 8px 12px; color: #f8fafc; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.15s ease; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); line-height: 1.4; }}
        
        /* ── Chart ── */
        .chart-panel{{ background:#0e1622; border:1px solid #1d2d44; border-radius:8px; padding:18px; margin-bottom:24px; }}
        .chart-header{{ display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:8px; border-bottom: 1px solid #1d2d44; padding-bottom: 8px; }}
        .chart-header h3{{ margin:0; font-size:15px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#94a3b8; }}
        .tf-buttons{{ display:flex; gap:6px; }}
        .tf-btn{{ background:#090e17; color:#64748b; border:1px solid #1d2d44; border-radius:4px; padding:5px 12px; cursor:pointer; font-size:11px; font-weight:700; transition:all .15s; text-transform:uppercase; }}
        .tf-btn:hover{{ background:#162235; color:#e2e8f0; border-color: #3b82f6; }}
        .tf-btn.active{{ background:#3b82f6; color:#fff; border-color:#3b82f6; }}
        .chart-info{{ margin-bottom:8px; }}
        .chart-info .eq-value{{ font-size:24px; font-weight:800; color:#f8fafc; }}
        .chart-info .eq-change{{ font-size:14px; margin-left:8px; font-weight:700; }}
        .chart-info .eq-date{{ display:block; color:#64748b; font-size:11px; margin-top:4px; text-transform: uppercase; font-weight: 500; letter-spacing: 0.05em; }}
        #equityCanvas{{ width:100%; height:260px; display:block; }}
        
        /* ── Table & Layout ── */
        .table-wrap{{ background:#0e1622; border:1px solid #1d2d44; border-radius:8px; overflow:hidden; margin-bottom:24px; }}
        table{{ width:100%; border-collapse:collapse; }}
        thead{{ background:#0a0f18; }}
        th,td{{ padding:12px 14px; border-bottom:1px solid #1d2d44; font-size:13px; text-align:left; white-space:nowrap; }}
        th{{ color:#cbd5e1; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; font-size:12px; }}
        tbody tr:hover{{ background:#162235; }}
        
        .note{{ color:#64748b; font-size:12px; font-style:italic; margin:8px 0 0; line-height: 1.4; }}
        
        @media(max-width:1024px){{
            .cards{{ grid-template-columns:repeat(2,minmax(180px,1fr)); }}
            .sections{{ grid-template-columns:1fr; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1 class="title">Trading Bot Performance Dashboard</h1>
            <p class="subtitle">Live Performance Report</p>
        </div>
        <div style="text-align: right; color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">
            Last Updated: <span style="color:#f8fafc;">{now_utc}</span>
        </div>
    </div>

    <!-- ── Top cards ── -->
    <div class="cards">
        <div class="card">
            <div class="card-label">Balance (Equity)</div>
            <div class="card-value">{fmt_money(account_metrics['equity'])}</div>
        </div>
        <div class="card">
            <div class="card-label">Sharpe Ratio</div>
            <div class="card-value text-blue">{risk_metrics['sharpe']:.2f}</div>
        </div>
        <div class="card">
            <div class="card-label">Max Drawdown</div>
            <div class="card-value {mdd_class}">{risk_metrics['max_drawdown_pct']:.2f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Daily P/L</div>
            <div class="card-value {daily_pl_class}">{fmt_money(account_metrics['daily_pl'])}</div>
        </div>
    </div>

    <!-- ── Equity chart ── -->
    <div class="chart-panel">
        <div class="chart-header">
            <h3>Portfolio Equity Trend</h3>
            <div class="tf-buttons">
                <button class="tf-btn active" data-tf="1D">1D</button>
                <button class="tf-btn" data-tf="1W">1W</button>
                <button class="tf-btn" data-tf="1M">1M</button>
                <button class="tf-btn" data-tf="1Y">1Y</button>
                <button class="tf-btn" data-tf="All">All</button>
            </div>
        </div>
        <div class="chart-info">
            <span class="eq-value" id="chartEqValue"></span>
            <span class="eq-change" id="chartEqChange"></span>
            <span class="eq-date" id="chartEqDate"></span>
        </div>
        <canvas id="equityCanvas"></canvas>
    </div>

    <!-- ── Three Column Visual Section ── -->
    <div class="sections">
        <!-- Col 1: Account Snapshot -->
        <div class="panel">
            <h3>Account Snapshot</h3>
            <div class="stats-grid">
                <div class="stat-box"><div class="label">Cash</div><div class="value">{fmt_money(account_metrics['cash'])}</div></div>
                <div class="stat-box"><div class="label">Last Close Equity</div><div class="value">{fmt_money(account_metrics['last_close_equity'])}</div></div>
                <div class="stat-box"><div class="label">Peak Equity (24H)</div><div class="value">{fmt_money(account_metrics['peak_equity_24h'])}</div></div>
                <div class="stat-box"><div class="label">All-Time Peak Equity</div><div class="value green">{fmt_money(account_metrics['all_time_peak'])}</div></div>
                <div class="stat-box full-width"><div class="label">Active Ongoing Trades</div><div class="value text-blue">{account_metrics['ongoing_trades']}</div></div>
            </div>
        </div>
        
        <!-- Col 2: Trade Statistics with Donut Chart -->
        <div class="panel">
            <h3>Trade Statistics</h3>
            <div class="trade-stats-container">
                <div class="donut-chart-wrap">
                    <svg class="donut-svg" viewBox="0 0 36 36">
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="#f43f5e" stroke-width="3"></circle>
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="#10b981" stroke-width="3"
                                stroke-dasharray="{win_dash} {loss_dash}" stroke-dashoffset="25"></circle>
                        <text x="18" y="16.5" class="donut-center-pct">{trade_stats['win_rate_pct']:.1f}%</text>
                        <text x="18" y="22.5" class="donut-center-sub">Win Rate</text>
                    </svg>
                </div>
                <div class="trade-stats-details">
                    <div class="stat-row">
                        <span class="stat-label">Winning Trades</span>
                        <span class="stat-val green">{trade_stats['win_trades']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Losing Trades</span>
                        <span class="stat-val red">{trade_stats['loss_trades']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Closed</span>
                        <span class="stat-val">{total_closed}</span>
                    </div>
                </div>
            </div>
            {trade_note}
        </div>

        <!-- Col 3: P&L Attribution Breakdown -->
        <div class="panel">
            <h3>P&amp;L Attribution (Gains)</h3>
            {attribution_html}
        </div>
    </div>

    <!-- ── Positions table ── -->
    <div class="table-wrap">
        <table style="width: 100%;">
            <thead><tr>
                <th>Symbol</th><th>Side</th><th>Qty</th>
                <th>Market Value</th><th>Avg Price</th><th>Current Price</th>
                <th>Unrealized P&amp;L ($)</th><th>Unrealized P&amp;L (%)</th><th>Today&#39;s Change</th>
            </tr></thead>
            <tbody>{position_rows}</tbody>
        </table>
    </div>

    <!-- ── Visual Bar Chart ── -->
    {bar_chart_html}

    <!-- ── Trade History Accordions ── -->
    {trade_history_html}

    <!-- ── Daily Performance Heatmap ── -->
    {equity_table_html}
</div>

<!-- ── Chart JS (vanilla, zero dependencies) ── -->
<script>
(function(){{
    var SERIES = {chart_data_json};
    var canvas = document.getElementById('equityCanvas');
    var ctx    = canvas.getContext('2d');
    var btns   = document.querySelectorAll('.tf-btn');
    var curTF  = '1D';

    function fmtMoney(v){{ return '$' + v.toFixed(2).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ','); }}
    function fmtPct(v){{ return (v>=0?'+':'')+v.toFixed(2)+'%'; }}
    function fmtDate(ms, tf){{
        var d=new Date(ms);
        if(tf==='1D') return d.toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}});
        return d.toLocaleDateString([],{{month:'short',day:'numeric',year:'numeric'}});
    }}

    function draw(tf){{
        var data=SERIES[tf]||[];
        var dpr=window.devicePixelRatio||1;
        var rect=canvas.getBoundingClientRect();
        canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
        ctx.scale(dpr,dpr);
        var W=rect.width, H=rect.height;

        // info
        var eqEl=document.getElementById('chartEqValue');
        var chEl=document.getElementById('chartEqChange');
        var dtEl=document.getElementById('chartEqDate');
        if(!data.length){{
            ctx.clearRect(0,0,W,H);
            ctx.fillStyle='#64748b'; ctx.font='14px sans-serif';
            ctx.fillText('No data for this timeframe',W/2-80,H/2);
            eqEl.textContent=''; chEl.textContent=''; dtEl.textContent='';
            return;
        }}
        var first=data[0][1], last=data[data.length-1][1];
        var pctChg=(last-first)/first*100;
        var positive=pctChg>=0;
        eqEl.textContent=fmtMoney(last);
        chEl.textContent=fmtPct(pctChg);
        chEl.className='eq-change '+(positive?'green':'red');
        dtEl.textContent=fmtDate(data[data.length-1][0],tf);

        var vals=data.map(function(p){{ return p[1]; }});
        var minV=Math.min.apply(null,vals), maxV=Math.max.apply(null,vals);
        var pad=10, range=maxV-minV||1;
        var yLabelW=60;
        var chartW=W-yLabelW-pad, chartH=H-pad*2;

        function xPos(i){{ return yLabelW+(i/(data.length-1||1))*chartW; }}
        function yPos(v){{ return pad+chartH-(((v-minV)/range)*chartH); }}

        ctx.clearRect(0,0,W,H);

        // grid lines + y labels
        ctx.strokeStyle='#1d2d44'; ctx.lineWidth=0.5;
        ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='right';
        var gridN=4;
        for(var g=0;g<=gridN;g++){{
            var gv=minV+(range/gridN)*g;
            var gy=yPos(gv);
            ctx.beginPath(); ctx.moveTo(yLabelW,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
            ctx.fillText(fmtMoney(gv),yLabelW-6,gy+4);
        }}

        // gradient fill
        var grad=ctx.createLinearGradient(0,pad,0,pad+chartH);
        if(positive){{
            grad.addColorStop(0,'rgba(16, 185, 129, 0.18)');
            grad.addColorStop(1,'rgba(16, 185, 129, 0.0)');
        }} else {{
            grad.addColorStop(0,'rgba(244, 63, 94, 0.18)');
            grad.addColorStop(1,'rgba(244, 63, 94, 0.0)');
        }}
        ctx.beginPath();
        ctx.moveTo(xPos(0),yPos(data[0][1]));
        for(var i=1;i<data.length;i++) ctx.lineTo(xPos(i),yPos(data[i][1]));
        ctx.lineTo(xPos(data.length-1),pad+chartH);
        ctx.lineTo(xPos(0),pad+chartH);
        ctx.closePath(); ctx.fillStyle=grad; ctx.fill();

        // line
        ctx.beginPath();
        ctx.moveTo(xPos(0),yPos(data[0][1]));
        for(var i=1;i<data.length;i++) ctx.lineTo(xPos(i),yPos(data[i][1]));
        ctx.strokeStyle=positive?'#10b981':'#f43f5e'; ctx.lineWidth=2; ctx.stroke();
    }}

    btns.forEach(function(b){{
        b.addEventListener('click',function(){{
            btns.forEach(function(x){{ x.classList.remove('active'); }});
            b.classList.add('active');
            curTF=b.getAttribute('data-tf');
            draw(curTF);
        }});
    }});

    window.addEventListener('resize',function(){{ draw(curTF); }});
    draw(curTF);
}})();

// ── Heatmap JS (vanilla, zero dependencies) ──
(function(){{
    var data = {heatmap_data_json};
    var container = document.getElementById('heatmapGrid');
    if(!container || !data.length) return;

    var dataMap = {{}};
    var maxVal = 0.1;
    data.forEach(function(d){{
        dataMap[d.date] = d;
        if(Math.abs(d.change_pct) > maxVal) {{
            maxVal = Math.abs(d.change_pct);
        }}
    }});

    var dates = data.map(function(d){{ return new Date(d.date + 'T00:00:00'); }});
    var minDate = new Date(Math.min.apply(null, dates));
    var maxDate = new Date(Math.max.apply(null, dates));

    // Sunday align
    var startDate = new Date(minDate);
    startDate.setDate(startDate.getDate() - startDate.getDay());

    // Saturday align
    var endDate = new Date(maxDate);
    endDate.setDate(endDate.getDate() + (6 - endDate.getDay()));

    var tempDate = new Date(startDate);
    var weeksCount = Math.ceil((endDate - startDate) / (7 * 24 * 60 * 60 * 1000));
    
    var colsHtml = '';
    var currentMonth = -1;
    var monthsLabelHtml = '<div class="heatmap-months-labels">';
    var weekColWidth = 14; 
    
    for (var w = 0; w < weeksCount; w++) {{
        var colHtml = '<div class="heatmap-week-col">';
        
        for (var d = 0; d < 7; d++) {{
            var dateStr = tempDate.toISOString().split('T')[0];
            var dayData = dataMap[dateStr];
            
            var cellStyle = '';
            var cellClass = 'heatmap-cell';
            var tooltipText = tempDate.toLocaleDateString([], {{month:'short', day:'numeric', year:'numeric'}}) + ' (No Trading)';
            
            if (dayData) {{
                var changePct = dayData.change_pct;
                var positive = changePct >= 0;
                var intensity = maxVal > 0 ? Math.min(1.0, Math.max(0.18, Math.abs(changePct) / maxVal)) : 0.5;
                
                if (changePct === 0) {{
                    cellStyle = 'background: #1e293b;';
                }} else if (positive) {{
                    cellStyle = 'background: rgba(16, 185, 129, ' + intensity.toFixed(2) + '); border: 1px solid rgba(16, 185, 129, ' + (intensity * 0.4).toFixed(2) + ');';
                }} else {{
                    cellStyle = 'background: rgba(244, 63, 94, ' + intensity.toFixed(2) + '); border: 1px solid rgba(244, 63, 94, ' + (intensity * 0.4).toFixed(2) + ');';
                }}
                
                tooltipText = '<strong>' + tempDate.toLocaleDateString([], {{month:'short', day:'numeric', year:'numeric'}}) + '</strong><br/>' +
                              'Equity: <strong>$' + dayData.equity.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}}) + '</strong><br/>' +
                              'Change: <strong class="' + (positive ? 'green' : 'red') + '">' + 
                              (positive ? '+' : '') + dayData.change.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}}) + 
                              ' (' + (positive ? '+' : '') + changePct.toFixed(2) + '%)</strong>';
                
                cellClass += ' active-cell';
            }} else {{
                cellClass += ' empty-cell';
            }}
            
            colHtml += '<div class="' + cellClass + '" style="' + cellStyle + '" data-tooltip="' + encodeURIComponent(tooltipText) + '"></div>';
            
            if (tempDate.getDay() === 0 && tempDate.getDate() <= 7) {{
                var m = tempDate.getMonth();
                if (m !== currentMonth) {{
                    currentMonth = m;
                    var monthName = tempDate.toLocaleDateString([], {{month:'short'}});
                    monthsLabelHtml += '<span style="position: absolute; left: ' + ((w * weekColWidth) + 34) + 'px;">' + monthName + '</span>';
                }}
            }}
            
            tempDate.setDate(tempDate.getDate() + 1);
        }}
        
        colHtml += '</div>';
        colsHtml += colHtml;
    }}
    
    monthsLabelHtml += '</div>';
    
    var dayLabelsHtml = '<div class="heatmap-day-labels">' +
                        '<span>Sun</span><span></span><span>Tue</span><span></span><span>Thu</span><span></span><span>Sat</span>' +
                        '</div>';
                        
    container.innerHTML = monthsLabelHtml + '<div class="heatmap-grid-inner">' + dayLabelsHtml + '<div class="heatmap-cols-wrap">' + colsHtml + '</div>' + '</div>';

    // Tooltips
    var tooltip = document.createElement('div');
    tooltip.className = 'heatmap-tooltip';
    document.body.appendChild(tooltip);
    
    var activeCells = container.querySelectorAll('.active-cell');
    activeCells.forEach(function(cell){{
        cell.addEventListener('mouseover', function(e){{
            var text = decodeURIComponent(cell.getAttribute('data-tooltip'));
            tooltip.innerHTML = text;
            tooltip.style.opacity = 1;
        }});
        cell.addEventListener('mousemove', function(e){{
            tooltip.style.left = (e.pageX + 12) + 'px';
            tooltip.style.top = (e.pageY - 20) + 'px';
        }});
        cell.addEventListener('mouseout', function(e){{
            tooltip.style.opacity = 0;
        }});
    }});
}})();
</script>
</body>
</html>"""
    return html


def main():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    account = client.get_account()

    equity = _safe_float(getattr(account, "equity", 0.0))
    cash = _safe_float(getattr(account, "cash", 0.0))
    last_close_equity = _safe_float(getattr(account, "last_equity", 0.0), default=equity)
    daily_pl = equity - last_close_equity

    chart_series = {}
    raw_histories = {}
    all_time_peak = equity  # fallback
    for cfg in CHART_TIMEFRAMES:
        try:
            hist = fetch_portfolio_history(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                paper=ALPACA_PAPER,
                period=cfg["period"],
                timeframe=cfg["timeframe"],
            )
            chart_series[cfg["key"]] = history_to_js_array(hist)
            raw_histories[cfg["key"]] = hist
            if cfg["key"] == "All" and not hist.empty:
                all_time_peak = float(hist["equity"].max())
        except Exception as exc:
            print(f"Warning: could not fetch {cfg['key']} history: {exc}")
            chart_series[cfg["key"]] = []
            raw_histories[cfg["key"]] = pd.DataFrame()

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
    trade_stats = calculate_trade_statistics(filled_orders)

    # ── Master rows & trade history calculation ───────────────────────
    roundtrips = build_roundtrips_from_orders(filled_orders, positions)
    
    master_rows = []
    for symbol, rts in roundtrips.items():
        pool = rts[0].get('pool', '—') if rts else '—'
        closed = [r for r in rts if r['pnl'] is not None]
        wins = [r for r in closed if r['pnl'] >= 0]
        losses = [r for r in closed if r['pnl'] < 0]
        stops = [r for r in closed if 'Stop' in r['exit_reason']]
        n_open = sum(1 for r in rts if r['pnl'] is None)
        total_pnl = sum(r['pnl'] for r in closed) if closed else 0.0
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

    trade_history_html = generate_trade_history_html(roundtrips, master_rows)

    account_metrics = {
        "equity": equity,
        "cash": cash,
        "last_close_equity": last_close_equity,
        "daily_pl": daily_pl,
        "peak_equity_24h": peak_equity_24h,
        "all_time_peak": all_time_peak,
        "ongoing_trades": ongoing_trades,
    }

    chart_data_json = json.dumps(chart_series)
    
    # ── Heatmap & Daily equity calculations ─────────────────────────────
    heatmap_data = []
    if not history_all_df.empty:
        df = history_all_df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['equity'] = df['equity'].astype(float)
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

    equity_table_html = generate_equity_table_html(history_all_df)

    html = generate_html(
        account_metrics, 
        risk_metrics, 
        trade_stats, 
        positions_df, 
        chart_data_json,
        trade_history_html,
        equity_table_html,
        heatmap_data_json,
        master_rows
    )

    output_path = os.path.join(REPORT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()