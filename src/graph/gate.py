from __future__ import annotations

import json
import statistics
from datetime import timedelta

from memory.schema import Event, WorkingMemory

# ---- config (tune these against scenario ground truth) ----------------------
# (source_system, event_type) pairs that are normally routine. Everything NOT
# listed here always wakes the agents.
ROUTINE_EVENTS = {
    ("web_app_events", "login"),
    ("card_payments", "purchase"),
}

HARD_STOP_KEYWORDS = {
    "legal": ["lawsuit", "legal action", "attorney", "lawyer", "sue you", "court"],
    "fraud": ["fraud", "unauthorized", "unauthorised", "hacked", "stolen", "scam"],
}

AMOUNT_KEYS = ("amount", "amount_inr", "value", "txn_amount")   # adjust to real payload
FAILURE_WORDS = ("fail", "declin", "denied", "blocked", "reject")

ABS_LARGE_AMOUNT = 50_000      # placeholder: set from the dataset's scale
SPIKE_RATIO = 3.0              # amount > 3x median of similar recent events
MIN_SAMPLES = 5                # need this many similar events for a median
BURST_COUNT = 8                # this many events...
BURST_WINDOW_MIN = 10          # ...within this many minutes
ACCUMULATION_EVERY = 15        # force a review after this many skipped events


def _text(payload: dict) -> str:
    return json.dumps(payload, default=str).lower()


def _amount(payload: dict) -> float | None:
    for k in AMOUNT_KEYS:
        v = payload.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.replace(",", ""))
            except ValueError:
                pass
    return None


def _decision(route: str, reasons: list[str], rule_ids: list[str]) -> dict:
    return {"route": route, "reasons": reasons, "rule_ids": rule_ids}


def evaluate_gate(event: Event, memory: WorkingMemory) -> dict:
    # Pure function: decides whether this LIVE event is worth waking agents for.
    text = _text(event.payload)
    key = (event.source_system, event.event_type)

    # 1) hard stops: deterministic, cannot be overridden by any LLM
    for category, words in HARD_STOP_KEYWORDS.items():
        hit = next((w for w in words if w in text), None)
        if hit:
            return _decision("hard_stop", [f"{category} keyword matched: '{hit}'"],
                             [f"HARD_STOP_{category.upper()}"])

    # 2) anything not known-routine always goes to the agents
    if key not in ROUTINE_EVENTS:
        return _decision("agents", [f"{key[0]}/{key[1]} is not a routine event type"],
                         ["NOT_ROUTINE"])

    # 3) routine event: wake agents only if something looks off
    reasons, rules = [], []
    others = [e for e in memory.recent_events if e.event_id != event.event_id]

    if any(w in text for w in FAILURE_WORDS):
        reasons.append("event indicates a failed/declined/blocked action")
        rules.append("FAILURE_SIGNAL")

    amt = _amount(event.payload)
    if amt is not None:
        if amt >= ABS_LARGE_AMOUNT:
            reasons.append(f"amount {amt:,.0f} >= absolute limit {ABS_LARGE_AMOUNT:,}")
            rules.append("LARGE_AMOUNT")
        similar = [a for e in others
                   if (e.source_system, e.event_type) == key
                   and (a := _amount(e.payload)) is not None]
        if len(similar) >= MIN_SAMPLES:
            med = statistics.median(similar)
            if med > 0 and amt > SPIKE_RATIO * med:
                reasons.append(f"amount {amt:,.0f} is {amt / med:.1f}x the recent median {med:,.0f}")
                rules.append("AMOUNT_SPIKE")

    window = timedelta(minutes=BURST_WINDOW_MIN)
    burst = sum(1 for e in others if abs(event.event_time - e.event_time) <= window)
    if burst >= BURST_COUNT:
        reasons.append(f"{burst} other events within {BURST_WINDOW_MIN} min (burst)")
        rules.append("BURST")

    if memory.events_since_analysis + 1 >= ACCUMULATION_EVERY:
        reasons.append(f"{memory.events_since_analysis + 1} routine events since last analysis (periodic review)")
        rules.append("PERIODIC_REVIEW")

    if rules:
        return _decision("agents", reasons, rules)
    return _decision("skip", [f"routine {key[0]}/{key[1]}, within normal range"], ["ROUTINE_OK"])