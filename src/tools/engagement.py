from collections import Counter

from tools.base import ToolInput, dedupe, in_window, pct, pfield, register

SRC = "app/web usage events (in-memory working memory, scoped)"
W = 14
LOGIN_HINTS = ("login", "session", "app_open", "launch")
TOPICS = {
    "home_loan": ("home loan", "mortgage", "housing", "property", "emi calculator"),
    "credit_card": ("credit card", "card offer", "limit increase"),
    "insurance": ("insurance", "policy", "cover"),
    "investment": ("mutual fund", "stock", "sip", "invest", "demat"),
    "education": ("education loan", "tuition", "student"),
    "travel_forex": ("forex", "travel", "international"),
    "personal_loan": ("personal loan", "instant loan"),
    "medical_hardship": ("medical", "hospital", "hardship", "payment plan", "payment arrangement", "health"),
    "debt_relief": ("debt", "late fee", "forbearance", "deferment", "waive"),
}


def _activity(inp: ToolInput):
    evs = dedupe(inp.events)
    logins = [e for e in evs if any(h in e.event_type.lower() for h in LOGIN_HINTS)]
    return (logins, "login/session events") if logins else (evs, "all in-scope usage events")


@register("engagement", reads=SRC)
def login_trend(inp: ToolInput) -> dict:
    """Activity count in the last 14d vs the 14d before, plus days since last activity."""
    ev, basis = _activity(inp)
    rec = sum(in_window(e, inp.now, W) for e in ev)
    prev = sum(in_window(e, inp.now, W, W) for e in ev)
    last = max((e.event_time for e in ev), default=None)
    return {"basis": basis, "window_days": W, "recent": rec, "previous": prev,
            "change_pct": pct(rec, prev), "days_since_last_activity": (inp.now - last).days if last else None}


@register("engagement", reads=SRC)
def feature_usage(inp: ToolInput) -> dict:
    """Which features are used now vs the previous window (stopped / newly used)."""
    rec, prev = Counter(), Counter()
    for e in dedupe(inp.events):
        f = pfield(e.payload, "feature")
        if f is None:
            continue
        if in_window(e, inp.now, W):
            rec[str(f)] += 1
        elif in_window(e, inp.now, W, W):
            prev[str(f)] += 1
    return {"recent_top": rec.most_common(5), "stopped_using": sorted(set(prev) - set(rec))[:5],
            "newly_used": sorted(set(rec) - set(prev))[:5]}


@register("engagement", reads=SRC)
def search_intent(inp: ToolInput) -> dict:
    """Counts of product-topic mentions (home loan, insurance, ...) in usage events of the last 14d."""
    hits = Counter()
    for e in dedupe(inp.events):
        if not in_window(e, inp.now, W):
            continue
        blob = f"{e.event_type} {e.payload}".lower()
        for topic, words in TOPICS.items():
            if any(w in blob for w in words):
                hits[topic] += 1
    return {"window_days": W, "topics": dict(hits)}

@register("engagement", reads=SRC)
def recent_searches(inp: ToolInput) -> dict:
    """The last few in-app search phrases (30d): direct customer intent, cited as evidence."""
    s = sorted((e.event_time, str(pfield(e.payload, "text"))) for e in dedupe(inp.events)
               if "search" in e.event_type.lower() and pfield(e.payload, "text") and in_window(e, inp.now, 30))
    return {"window_days": 30,
            "searches": [{"date": t.date().isoformat(), "text": x[:120]} for t, x in s[-5:]]}