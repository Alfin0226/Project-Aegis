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

def generate_trade_history_html(roundtrips):
    """Generate dark-themed HTML tables for each stock's roundtrip trades + overall summary."""
    css = """
    <style>
        .th-section { margin-top: 32px; }
        .th-header { font-size: 20px; font-weight: 600; margin-bottom: 16px; border-bottom: 2px solid #1e2a44; padding-bottom: 8px; }
        .th-symbol-header { font-size: 18px; margin-top: 24px; color: #3b82f6; border-left: 4px solid #3b82f6; padding-left: 10px; }
        .th-pool-label { font-size: 14px; color: #94a3b8; margin: 4px 0 12px 0; font-weight: normal; }
        .th-table-wrap { background: #111b2e; border: 1px solid #1e2a44; border-radius: 8px; overflow: hidden; margin-bottom: 16px; }
        .th-table { width: 100%; border-collapse: collapse; }
        .th-table th { background: #0f172a; color: #cbd5e1; font-weight: 600; padding: 10px; text-align: left; font-size: 12px; white-space: nowrap; border-bottom: 1px solid #1e2a44; }
        .th-table td { padding: 8px 10px; border-bottom: 1px solid #1e2a44; font-size: 13px; white-space: nowrap; }
        .th-table tbody tr:hover { background: #1e293b; }
        .pnl-pos { color: #22c55e; font-weight: 600; }
        .pnl-neg { color: #ef4444; font-weight: 600; }
        .pnl-open { color: #94a3b8; font-style: italic; }
        .reason-badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; display: inline-block; }
        .reason-stop { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
        .reason-exit { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
        .reason-open { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }
        .summary-wrap { background: #0f172a; padding: 10px; border-top: 1px solid #1e2a44; }
        .summary-table { width: 100%; border-collapse: collapse; }
        .summary-table th { color: #94a3b8; font-size: 11px; text-align: center; padding: 4px; border: none; background: transparent; }
        .summary-table td { text-align: center; font-size: 13px; font-weight: 500; padding: 4px; border: none; }
        .master-summary { margin-top: 40px; }
        .master-table th { background: #1e293b; color: #f8fafc; font-size: 12px; }
        .total-row td { background: #1e293b; font-weight: bold; border-top: 2px solid #334155; }
    </style>
    """

    def fmt_date(d):
        if not d: return '—'
        if isinstance(d, datetime):
            return d.strftime('%Y-%m-%d')
        # Handle string dates from alpaca if any
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

    master_rows = []

    for symbol in sorted(roundtrips.keys()):
        rts = roundtrips[symbol]
        pool = rts[0].get('pool', '—') if rts else '—'

        html_parts.append(f'<div class="th-symbol-header">{symbol}</div>')
        html_parts.append(f'<div class="th-pool-label">Pool: {pool} &mdash; {len(rts)} roundtrip(s)</div>')
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
        n_open = sum(1 for r in rts if r['pnl'] is None)

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

        html_parts.append('</tbody></table>')

        total_pnl = sum(r['pnl'] for r in closed) if closed else 0
        win_rate = (len(wins) / len(closed) * 100) if closed else 0
        avg_win = sum(r['pnl'] for r in wins)/len(wins) if wins else 0
        avg_loss = sum(r['pnl'] for r in losses)/len(losses) if losses else 0

        html_parts.append('<div class="summary-wrap"><table class="summary-table">')
        html_parts.append(
            '<thead><tr><th>Closed</th><th>Wins</th><th>Losses</th>'
            '<th>Stops</th><th>Open</th><th>Win Rate</th>'
            '<th>Avg Win</th><th>Avg Loss</th><th>Net P&amp;L</th></tr></thead><tbody>'
        )
        wr_cls = 'pnl-pos' if win_rate >= 50 else 'pnl-neg'
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
        html_parts.append('</tbody></table></div></div>')

        master_rows.append({
            'symbol': symbol, 'pool': pool,
            'total': len(rts), 'closed': len(closed),
            'wins': len(wins), 'losses': len(losses),
            'stops': len(stops), 'open': n_open,
            'win_rate': win_rate, 'total_pnl': total_pnl,
        })

    # --- Master Summary ---
    html_parts.append('<div class="master-summary">')
    html_parts.append('<div class="th-header">Overall Stock Summary</div>')
    html_parts.append('<div class="th-table-wrap"><table class="th-table master-table">')
    html_parts.append(
        '<thead><tr><th>Stock</th><th>Pool</th><th>Total</th><th>Closed</th>'
        '<th>Wins</th><th>Losses</th><th>Stops</th><th>Open</th>'
        '<th>Win Rate</th><th>Net P&amp;L</th></tr></thead><tbody>'
    )

    grand_total = grand_closed = grand_wins = grand_losses = grand_stops = grand_open = 0
    grand_pnl = 0.0

    for row in sorted(master_rows, key=lambda r: r['total_pnl'], reverse=True):
        wr_cls = 'pnl-pos' if row['win_rate'] >= 50 else 'pnl-neg'
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
    """Generate a day-to-day equity value HTML table from the daily history DataFrame."""
    if daily_history_df.empty or len(daily_history_df) < 2:
        return '<div class="panel" style="margin-top:16px;"><h3>Daily Equity History</h3><p class="note">Not enough data to display daily equity history.</p></div>'

    df = daily_history_df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['equity'] = df['equity'].astype(float)

    # Calculate daily change and cumulative return
    df['daily_change'] = df['equity'].diff()
    df['daily_change_pct'] = df['equity'].pct_change() * 100
    first_equity = df['equity'].iloc[0]
    df['cumulative_return'] = ((df['equity'] / first_equity) - 1) * 100

    # Build rows (newest first)
    rows_html = ''
    for _, row in df.iloc[::-1].iterrows():
        date_str = row['timestamp'].strftime('%Y-%m-%d')
        eq_str = f"${row['equity']:,.2f}"

        if pd.isna(row['daily_change']):
            chg_str = '—'
            chg_pct_str = '—'
            chg_cls = ''
        else:
            chg = row['daily_change']
            chg_pct = row['daily_change_pct']
            chg_cls = 'pnl-pos' if chg >= 0 else 'pnl-neg'
            sign = '+' if chg >= 0 else ''
            chg_str = f'{sign}${chg:,.2f}'
            chg_pct_str = f'{sign}{chg_pct:.2f}%'

        cum_ret = row['cumulative_return']
        cum_cls = 'pnl-pos' if cum_ret >= 0 else 'pnl-neg'
        cum_sign = '+' if cum_ret >= 0 else ''
        cum_str = f'{cum_sign}{cum_ret:.2f}%'

        rows_html += f'''<tr>
            <td>{date_str}</td>
            <td>{eq_str}</td>
            <td class="{chg_cls}">{chg_str}</td>
            <td class="{chg_cls}">{chg_pct_str}</td>
            <td class="{cum_cls}">{cum_str}</td>
        </tr>\n'''

    html = f'''
    <div class="equity-history-section" style="margin-top:32px;">
        <div class="th-header">Daily Equity History</div>
        <div class="th-table-wrap">
            <table class="th-table" id="equityHistoryTable">
                <thead><tr>
                    <th>Date</th>
                    <th>Equity</th>
                    <th>Daily Change ($)</th>
                    <th>Daily Change (%)</th>
                    <th>Cumulative Return</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
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


def generate_html(account_metrics, risk_metrics, trade_stats, positions_df, chart_data_json, trade_history_html, equity_table_html):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    daily_pl_class = "green" if account_metrics["daily_pl"] >= 0 else "red"
    mdd_class = "green" if risk_metrics["max_drawdown_pct"] >= -5 else "red"

    # Trade stats explanation
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

    # Position rows
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
            position_rows += f"""
            <tr>
                <td>{row['symbol']}</td>
                <td>{row['side']}</td>
                <td>{row['qty']:,.4f}</td>
                <td>{fmt_money(row['market_value'])}</td>
                <td>{fmt_money(row['avg_entry_price'])}</td>
                <td>{fmt_money(row['current_price'])}</td>
                <td class="{pl_class}">{fmt_money(row['unrealized_pl'])}</td>
                <td class="{pl_class}">{fmt_pct(row['unrealized_plpc'])}</td>
                <td class="{change_class}">{fmt_pct(row['change_today'])}</td>
            </tr>
            """

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
        body{{ margin:0; background:#0b1220; color:#e2e8f0; font-family:Inter,Segoe UI,Arial,sans-serif; }}
        .container{{ max-width:1280px; margin:0 auto; padding:24px; }}
        .header{{ margin-bottom:20px; }}
        .title{{ margin:0; font-size:28px; }}
        .subtitle{{ margin:6px 0 0; color:#94a3b8; font-size:14px; }}
        .cards{{ display:grid; grid-template-columns:repeat(4,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }}
        .card{{ background:#111b2e; border:1px solid #1e2a44; border-radius:10px; padding:14px; }}
        .label{{ color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
        .value{{ margin-top:6px; font-size:24px; font-weight:700; color:#f8fafc; }}
        .green{{ color:#22c55e; }} .red{{ color:#ef4444; }} .yellow{{ color:#eab308; }}
        /* ── Chart ── */
        .chart-panel{{ background:#111b2e; border:1px solid #1e2a44; border-radius:10px; padding:16px; margin-bottom:16px; }}
        .chart-header{{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px; }}
        .chart-header h3{{ margin:0; font-size:16px; }}
        .tf-buttons{{ display:flex; gap:4px; }}
        .tf-btn{{ background:#1e293b; color:#94a3b8; border:1px solid #334155; border-radius:6px; padding:5px 12px; cursor:pointer; font-size:12px; font-weight:600; transition:all .15s; }}
        .tf-btn:hover{{ background:#334155; color:#e2e8f0; }}
        .tf-btn.active{{ background:#3b82f6; color:#fff; border-color:#3b82f6; }}
        .chart-info{{ margin-bottom:8px; }}
        .chart-info .eq-value{{ font-size:22px; font-weight:700; }}
        .chart-info .eq-change{{ font-size:14px; margin-left:8px; }}
        .chart-info .eq-date{{ display:block; color:#94a3b8; font-size:12px; margin-top:2px; }}
        #equityCanvas{{ width:100%; height:260px; display:block; }}
        /* ── Sections ── */
        .sections{{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; }}
        .panel{{ background:#111b2e; border:1px solid #1e2a44; border-radius:10px; padding:14px; }}
        .panel h3{{ margin:0 0 12px; font-size:16px; }}
        .stats-grid{{ display:grid; grid-template-columns:repeat(2,minmax(100px,1fr)); gap:10px; }}
        .stat-box{{ background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:10px; }}
        .note{{ color:#94a3b8; font-size:12px; font-style:italic; margin:8px 0 0; }}
        /* ── Table ── */
        .table-wrap{{ background:#111b2e; border:1px solid #1e2a44; border-radius:10px; overflow:hidden; }}
        table{{ width:100%; border-collapse:collapse; }}
        thead{{ background:#0f172a; }}
        th,td{{ padding:11px 10px; border-bottom:1px solid #1e2a44; font-size:13px; text-align:left; white-space:nowrap; }}
        th{{ color:#cbd5e1; font-weight:600; }}
        tbody tr:hover{{ background:#0f172a; }}
        @media(max-width:1024px){{
            .cards{{ grid-template-columns:repeat(2,minmax(180px,1fr)); }}
            .sections{{ grid-template-columns:1fr; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1 class="title">Trading Bot Performance Dashboard</h1>
        <p class="subtitle">Last Updated: {now_utc}</p>
    </div>

    <!-- ── Top cards ── -->
    <div class="cards">
        <div class="card"><div class="label">Balance (Equity)</div><div class="value">{fmt_money(account_metrics['equity'])}</div></div>
        <div class="card"><div class="label">Sharpe Ratio</div><div class="value">{risk_metrics['sharpe']:.2f}</div></div>
        <div class="card"><div class="label">Max Drawdown</div><div class="value {mdd_class}">{risk_metrics['max_drawdown_pct']:.2f}%</div></div>
        <div class="card"><div class="label">Daily P/L</div><div class="value {daily_pl_class}">{fmt_money(account_metrics['daily_pl'])}</div></div>
    </div>

    <!-- ── Equity chart ── -->
    <div class="chart-panel">
        <div class="chart-header">
            <h3>Portfolio Equity</h3>
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

    <!-- ── Account snapshot + Trade stats ── -->
    <div class="sections">
        <div class="panel">
            <h3>Account Snapshot</h3>
            <div class="stats-grid">
                <div class="stat-box"><div class="label">Cash</div><div class="value">{fmt_money(account_metrics['cash'])}</div></div>
                <div class="stat-box"><div class="label">Last Close Equity</div><div class="value">{fmt_money(account_metrics['last_close_equity'])}</div></div>
                <div class="stat-box"><div class="label">Peak Equity (24H)</div><div class="value">{fmt_money(account_metrics['peak_equity_24h'])}</div></div>
                <div class="stat-box"><div class="label">All-Time Peak Equity</div><div class="value green">{fmt_money(account_metrics['all_time_peak'])}</div></div>
                <div class="stat-box"><div class="label">Ongoing Trades</div><div class="value">{account_metrics['ongoing_trades']}</div></div>
            </div>
        </div>
        <div class="panel">
            <h3>Trade Statistics</h3>
            <div class="stats-grid">
                <div class="stat-box"><div class="label">Winning Trades</div><div class="value green">{trade_stats['win_trades']}</div></div>
                <div class="stat-box"><div class="label">Losing Trades</div><div class="value red">{trade_stats['loss_trades']}</div></div>
                <div class="stat-box"><div class="label">Win Rate</div><div class="value">{trade_stats['win_rate_pct']:.2f}%</div></div>
                <div class="stat-box"><div class="label">Closed Trades</div><div class="value">{total_closed}</div></div>
            </div>
            {trade_note}
        </div>
    </div>

    <!-- ── Positions table ── -->
    <div class="table-wrap">
        <table>
            <thead><tr>
                <th>Symbol</th><th>Side</th><th>Qty</th>
                <th>Size (Market Value)</th><th>Average Buy Price</th><th>Current Price</th>
                <th>Unrealized P&amp;L ($)</th><th>P&amp;L (%)</th><th>Today&#39;s Change (%)</th>
            </tr></thead>
            <tbody>{position_rows}</tbody>
        </table>
    </div>

    <!-- ── Trade History ── -->
    {trade_history_html}

    <!-- ── Daily Equity History Table ── -->
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
            ctx.fillStyle='#94a3b8'; ctx.font='14px sans-serif';
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
        // y-axis label width
        var yLabelW=60;
        var chartW=W-yLabelW-pad, chartH=H-pad*2;

        function xPos(i){{ return yLabelW+(i/(data.length-1||1))*chartW; }}
        function yPos(v){{ return pad+chartH-(((v-minV)/range)*chartH); }}

        ctx.clearRect(0,0,W,H);

        // grid lines + y labels
        ctx.strokeStyle='#1e2a44'; ctx.lineWidth=0.5;
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
            grad.addColorStop(0,'rgba(34,197,94,0.25)');
            grad.addColorStop(1,'rgba(34,197,94,0.0)');
        }} else {{
            grad.addColorStop(0,'rgba(239,68,68,0.25)');
            grad.addColorStop(1,'rgba(239,68,68,0.0)');
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
        ctx.strokeStyle=positive?'#22c55e':'#ef4444'; ctx.lineWidth=2; ctx.stroke();
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

    # ── Fetch all chart timeframes (single pass) ─────────────────────
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

    # Reuse already-fetched data for risk metrics
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
    # Fetch ALL closed orders using pagination (Alpaca limits to 500 per request)
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
        # Slight negative offset to avoid fetching the exact same order again
        until_time = dt

    filled_orders = [
        order
        for order in closed_orders
        if _safe_float(getattr(order, "filled_qty", 0.0)) > 0
        and _safe_float(getattr(order, "filled_avg_price", 0.0)) > 0
    ]
    trade_stats = calculate_trade_statistics(filled_orders)

    # ── Per-Stock Trade History & Roundtrips ───────────────────────────
    roundtrips = build_roundtrips_from_orders(filled_orders, positions)
    trade_history_html = generate_trade_history_html(roundtrips)

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
    # ── Daily Equity Table ──────────────────────────────────────────────
    all_daily_history = raw_histories.get("All", pd.DataFrame())
    equity_table_html = generate_equity_table_html(all_daily_history)

    html = generate_html(
        account_metrics, 
        risk_metrics, 
        trade_stats, 
        positions_df, 
        chart_data_json,
        trade_history_html,
        equity_table_html
    )

    output_path = os.path.join(REPORT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()