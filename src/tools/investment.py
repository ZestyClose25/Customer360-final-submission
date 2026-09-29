from datetime import timedelta

from tools.base import ToolInput, dedupe, in_window, num, pct, pfield, r2, register

SRC = "brokerage / portfolio events (in-memory working memory, scoped)"
TRADE_HINTS = ("trade", "buy", "sell", "order", "redeem", "withdraw")


def _latest(inp, name):
    return sorted([(e.event_time, num(pfield(e.payload, name))) for e in dedupe(inp.events)
                   if num(pfield(e.payload, name)) is not None])


def in_window_t(t, now, days):
    return now - timedelta(days=days) < t <= now


@register("investment", reads=SRC)
def portfolio_snapshot(inp: ToolInput) -> dict:
    """Latest portfolio value and change vs the reading ~30d earlier (or the oldest one)."""
    v = _latest(inp, "portfolio_value")
    if not v:
        return {"has_portfolio_events": False}
    old = [x for t, x in v if not in_window_t(t, inp.now, 30)]
    base = old[-1] if old else v[0][1]
    return {"has_portfolio_events": True, "latest_value": r2(v[-1][1]), "change_pct_vs_30d": pct(v[-1][1], base)}


@register("investment", reads=SRC)
def trade_activity(inp: ToolInput) -> dict:
    """Trades in the last 30d: count, buys vs sells, total and largest notional."""
    tr = [e for e in dedupe(inp.events) if any(h in e.event_type.lower() for h in TRADE_HINTS)
          and in_window(e, inp.now, 30)]
    amts = [abs(num(pfield(e.payload, "amount")) or 0) for e in tr]
    return {"n_trades_30d": len(tr),
            "sells_or_withdrawals": sum(any(w in e.event_type.lower() for w in ("sell", "redeem", "withdraw")) for e in tr),
            "total_notional": r2(sum(amts)), "largest": r2(max(amts, default=0))}


@register("investment", reads=SRC)
def idle_cash(inp: ToolInput) -> dict:
    """Latest uninvested cash and its share of the portfolio (offer opportunity signal)."""
    c, p = _latest(inp, "cash"), _latest(inp, "portfolio_value")
    if not c:
        return {"has_cash_events": False}
    total = (p[-1][1] + c[-1][1]) if p else None
    return {"has_cash_events": True, "idle_cash": r2(c[-1][1]),
            "share_of_total_pct": round(c[-1][1] / total * 100, 1) if total else None}