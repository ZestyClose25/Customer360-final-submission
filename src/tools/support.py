import re
from collections import Counter

from tools.base import ToolInput, dedupe, in_window, pfield, register

SRC = "support tickets / chat / call events (in-memory working memory, scoped)"
RESOLVED = {"resolved", "closed", "done", "completed", "solved"}
NEG = {"angry", "furious", "unacceptable", "worst", "terrible", "useless", "disappointed", "frustrated",
       "horrible", "ridiculous", "waste", "unhappy", "complaint", "pathetic", "disgusted", "scam"}
POS = {"thanks", "thank", "great", "happy", "resolved", "helpful", "appreciate", "excellent", "good", "love"}
# shared with the guardrail node later: from tools.support import URGENCY_KEYWORDS
URGENCY_KEYWORDS = {
    "legal": ["lawsuit", "legal action", "attorney", "lawyer", "sue ", "court", "consumer forum"],
    "churn": ["cancel", "close my account", "closing my account", "switch bank", "leaving", "transfer out"],
    "fraud": ["fraud", "unauthorized", "unauthorised", "stolen", "hacked", "did not make this", "scam"],
    "self_harm": ["suicide", "kill myself", "end my life", "self-harm"],
    "hardship": ["payment plan", "payment arrangement", "can't pay", "cannot pay", "hardship", "income dropped",
                 "lost my job", "laid off", "behind on", "late fee", "forbearance", "deferment"],
    "medical": ["hospital", "surgery", "diagnosis", "medical", "illness", "treatment", "er visit"],
}


def _text(e) -> str:
    t = pfield(e.payload, "text")
    return str(t) if t is not None else ""


def _score(text: str) -> float:
    w = re.findall(r"[a-z']+", text.lower())
    neg, pos = sum(x in NEG for x in w), sum(x in POS for x in w)
    return round((pos - neg) / max(1, pos + neg), 2)


def _label(s: float) -> str:
    return "negative" if s <= -0.4 else "positive" if s >= 0.4 else "neutral"


@register("support", reads=SRC)
def ticket_history(inp: ToolInput) -> dict:
    """Ticket counts, how many are still open, and the age of the oldest open one."""
    evs = dedupe(inp.events)
    open_ = [e for e in evs if str(pfield(e.payload, "status", "open")).lower() not in RESOLVED]
    return {"n_tickets": len(evs), "n_last_30d": sum(in_window(e, inp.now, 30) for e in evs),
            "n_open": len(open_),
            "open_categories": sorted({str(pfield(e.payload, "category")) for e in open_ if pfield(e.payload, "category")}),
            "open_channels": sorted({str(pfield(e.payload, "channel")) for e in open_ if pfield(e.payload, "channel")}),
            "oldest_open_age_days": max(((inp.now - e.event_time).days for e in open_), default=None)}


@register("support", reads=SRC)
def score_sentiment(inp: ToolInput) -> dict:
    """Lexicon sentiment (-1..1) of the triggering ticket and the average of the last 3."""
    evs = sorted(dedupe(inp.events), key=lambda e: e.event_time)[-3:]
    scores = [_score(_text(e)) for e in evs if _text(e)]
    t = _score(_text(inp.trigger)) if inp.trigger and _text(inp.trigger) else None
    avg = round(sum(scores) / len(scores), 2) if scores else None
    return {"trigger_score": t, "trigger_label": _label(t) if t is not None else None,
            "last3_avg": avg, "last3_label": _label(avg) if avg is not None else None}


@register("support", reads=SRC)
def scan_urgency_keywords(inp: ToolInput) -> dict:
    """Keyword scan of the triggering text for legal / churn / fraud / self-harm language."""
    txt = _text(inp.trigger).lower() if inp.trigger else ""
    if not txt:
        return {"applicable": False}
    hits = {k: [w.strip() for w in ws if w in txt] for k, ws in URGENCY_KEYWORDS.items()}
    hits = {k: v for k, v in hits.items() if v}
    return {"applicable": True, "hits": hits,
            "hard_stop_candidate": any(k in hits for k in ("legal", "fraud", "self_harm"))}


@register("support", reads=SRC)
def repeat_contacts(inp: ToolInput) -> dict:
    """How many times the customer contacted support in 30d, by event type."""
    evs = [e for e in dedupe(inp.events) if in_window(e, inp.now, 30)]
    return {"contacts_30d": len(evs), "by_type": dict(Counter(e.event_type for e in evs))}