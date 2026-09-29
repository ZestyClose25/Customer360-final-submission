from collections import defaultdict
from datetime import timedelta
from statistics import mean, median, pstdev

from tools.base import (INCOME_KINDS, INTERNAL_KINDS, ToolInput, dedupe, in_window, is_credit, num, pct,
                        pfield, r2, register)

SRC = "card / ledger / wire events (in-memory working memory, scoped)"


def _amt(e):
    a = num(pfield(e.payload, "amount"))
    return abs(a) if a is not None else None


def _kind(e) -> str:
    return str(pfield(e.payload, "txn_kind", "")).lower()


def _span(pairs, now, days, offset=0):
    lo, hi = now - timedelta(days=days + offset), now - timedelta(days=offset)
    return sum(a for e, a in pairs if lo < e.event_time <= hi)


def _purchases(inp):
    return [(e, _amt(e)) for e in dedupe(inp.events) if e.event_type in ("purchase", "payment") and _amt(e) is not None]


@register("transaction", reads=SRC)
def spend_baseline(inp: ToolInput) -> dict:
    """Card-spend baseline (triggering purchase excluded): stats, 30d vs prev 30d, category shifts, refunds."""
    trig = inp.trigger.event_id if inp.trigger else None
    hist = [(e, a) for e, a in _purchases(inp) if e.event_id != trig]
    if not hist:
        return {"n_history": 0, "note": "no card purchase history"}
    amts = [a for _, a in hist]
    cur, prev = _span(hist, inp.now, 30), _span(hist, inp.now, 30, 30)
    cats_cur, cats_prev = defaultdict(float), defaultdict(float)
    for e, a in hist:
        c = str(pfield(e.payload, "category", "unknown"))
        if in_window(e, inp.now, 30):
            cats_cur[c] += a
        elif in_window(e, inp.now, 30, 30):
            cats_prev[c] += a
    floor = 10 * median(amts)          # a category only "surges" if it is >=3x before AND material for this customer
    surging = sorted(c for c, v in cats_cur.items() if v >= 3 * max(cats_prev.get(c, 0), 100) and v >= floor)
    refunds = sum(_amt(e) or 0 for e in dedupe(inp.events) if e.event_type == "refund" and in_window(e, inp.now, 30))
    return {"n_history": len(amts), "mean": r2(mean(amts)), "median": r2(median(amts)), "std": r2(pstdev(amts)),
            "spend_last_30d": r2(cur), "spend_prev_30d": r2(prev), "change_pct": pct(cur, prev),
            "top_categories_30d": sorted(((c, r2(v)) for c, v in cats_cur.items()), key=lambda x: -x[1])[:5],
            "surging_categories": surging, "refunds_30d": r2(refunds)}


@register("transaction", reads=SRC)
def txn_anomaly(inp: ToolInput) -> dict:
    """Score the triggering event against this customer's history of the SAME kind (event_type + ledger type)."""
    t = inp.trigger
    a = _amt(t) if t else None
    if a is None:
        return {"applicable": False}
    grp = (t.event_type, _kind(t))
    hist = [(e, _amt(e)) for e in dedupe(inp.events)
            if e.event_id != t.event_id and (e.event_type, _kind(e)) == grp and _amt(e) is not None]
    amts = [x for _, x in hist]
    res = {"applicable": True, "kind": "/".join(x for x in grp if x), "amount": a,
           "direction": "credit" if is_credit(t) else "debit", "n_history": len(amts), "flags": []}
    if _kind(t) in INTERNAL_KINDS:
        res["flags"].append("internal_transfer")             # own-account move: neither spend nor income
    if pfield(t.payload, "intl") is True:
        res["flags"].append("international")
    bal = num(pfield(t.payload, "balance"))
    if bal is not None and not is_credit(t):
        res["share_of_balance_pct"] = round(a / (bal + a) * 100, 1)       # balance_after is post-debit
    if len(amts) < 5:
        res["note"] = "fewer than 5 comparable events; no statistical flags"
        return res
    m, sd, med = mean(amts), pstdev(amts), median(amts)
    z = (a - m) / sd if sd > 0 else None
    res.update(z_score=r2(z), x_median=r2(a / med) if med else None)
    if (z is not None and z >= 2.5) or (med and a / med >= 5):
        res["flags"].append("amount_outlier")
    for name in ("merchant", "category", "country"):
        v = pfield(t.payload, name)
        if v is not None and str(v) not in {str(pfield(e.payload, name)) for e, _ in hist}:
            res["flags"].append(f"new_{name}")
    return res


@register("transaction", reads=SRC)
def income_pattern(inp: ToolInput) -> dict:
    """Real income only (salary / benefits credits): 30d vs prev 30d, income-type switch, salary gap."""
    inc = sorted([(e, _amt(e)) for e in dedupe(inp.events)
                  if _kind(e) in INCOME_KINDS and _amt(e) is not None], key=lambda p: p[0].event_time)
    if not inc:
        return {"n_income_events": 0, "note": "no salary/benefit credits seen"}
    now = inp.now
    cur, prev = _span(inc, now, 30), _span(inc, now, 30, 30)
    k_now = sorted({_kind(e) for e, _ in inc if in_window(e, now, 30)})
    k_prev = sorted({_kind(e) for e, _ in inc if in_window(e, now, 30, 30)})
    sal = [e for e, _ in inc if _kind(e) == "salary_credit"]
    last_e, last_a = inc[-1]
    others = [a for _, a in inc[:-1]]
    gaps = [(b.event_time - a.event_time).days for a, b in zip(sal, sal[1:])]
    gap = median(gaps) if gaps else None
    since_sal = (now - sal[-1].event_time).days if sal else None
    return {"n_income_events": len(inc), "income_30d": r2(cur), "income_prev_30d": r2(prev),
            "change_pct": pct(cur, prev), "last_income_kind": _kind(last_e), "last_income_amount": r2(last_a),
            "last_income_date": last_e.event_time.date().isoformat(),
            "amount_vs_typical_pct": pct(last_a, median(others)) if others else None,
            "kinds_last_30d": k_now, "kinds_prev_30d": k_prev,
            "income_type_changed": bool(k_now and k_prev and set(k_now) != set(k_prev)),
            "days_since_last_salary": since_sal,
            "salary_overdue": bool(gap and since_sal is not None and since_sal > 1.5 * gap)}


@register("transaction", reads=SRC)
def balance_trend(inp: ToolInput) -> dict:
    """Per-account balance (ledger balance_after): latest, change vs ~30d earlier, 30d low."""
    by_acc = defaultdict(list)
    for e in dedupe(inp.events):
        b = num(pfield(e.payload, "balance"))
        if b is not None and e.account_id:
            by_acc[e.account_id].append((e.event_time, b))
    if not by_acc:
        return {"has_balance_events": False}
    out, total = {}, 0.0
    for acc, v in by_acc.items():
        v.sort()
        old = [b for t, b in v if t <= inp.now - timedelta(days=30)]
        ref = old[-1] if old else v[0][1]
        last30 = [b for t, b in v if t > inp.now - timedelta(days=30)] or [v[-1][1]]
        out[acc] = {"latest": r2(v[-1][1]), "change_pct_vs_30d": pct(v[-1][1], ref), "low_30d": r2(min(last30))}
        total += v[-1][1]
    return {"has_balance_events": True, "accounts": out, "total_latest": r2(total)}