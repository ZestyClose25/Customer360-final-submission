import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from observe import emit

# payload field aliases
ALIASES = {
    "amount":    ["amount", "txn_amount", "transaction_amount", "value", "amt"],
    "direction": ["direction", "txn_type", "credit_debit", "dr_cr"],
    "txn_kind":  ["transaction_type"],                     # ledger: salary_credit, mortgage_payment, ...
    "category":  ["category", "mcc_category", "merchant_category", "mcc", "merchant_type"],
    "merchant":  ["merchant_name", "merchant", "counterparty_name", "payee", "counterparty"],
    "country":   ["counterparty_country", "country", "merchant_country", "location_country"],
    "text":      ["raw_text", "search_text", "text", "message", "body", "description", "transcript",
                  "summary", "subject", "notes"],
    "status":    ["status", "resolution_status", "ticket_status", "state"],
    "device":    ["device_type", "device", "device_id"],
    "feature":   ["feature_or_page", "feature", "screen", "page", "module"],
    "balance":   ["balance_after", "balance"],
    "intl":      ["is_international"],
    "expiry":    ["expiry", "expiry_date", "valid_until"],
    "portfolio_value": ["portfolio_value", "holdings_value", "total_value", "nav"],
    "cash":      ["cash_balance", "idle_cash", "cash"],
}
CREDIT_VALUES = {"credit", "cr", "c", "deposit", "inflow", "inbound", "in"}
INCOME_KINDS = {"salary_credit", "benefits_credit", "pension_credit"}      # real income
INTERNAL_KINDS = {"internal_transfer_in", "internal_transfer_out"}

def pfield(payload: dict | None, name: str, default=None):
    p = payload or {}
    for k in ALIASES.get(name, [name]):
        if k in p and p[k] not in (None, ""):
            return p[k]
    return default

def num(x) -> float | None:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None

def r2(x):
    return None if x is None else round(x, 2)

def pct(new, old):
    return None if not old else round((new-old) / old * 100, 1)

def is_credit(e) -> bool:
    d = str(pfield(e.payload, "direction", "")).lower()
    et = e.event_type.lower()
    k = str(pfield(e.payload, "txn_kind", "")).lower()
    return (d in CREDIT_VALUES or k.endswith("_credit") or k.endswith("_transfer_in")
            or any(w in et for w in ("credit", "deposit", "salary", "refund")))

def replay_now(events, fallback: datetime) -> datetime:
    """Replay clock = newest event_time seen so far. A late event (event_time in the past, ingested
    later) must not rewind the clock, or newer events would fall outside every window."""
    ts = [e.event_time for e in events]
    return max(ts) if ts else fallback

def dedupe(events) -> list:
    seen, out = set(), []
    for e in events:
        if e.event_id not in seen:
            seen.add(e.event_id)
            out.append(e)
    return out

def in_window(e, now: datetime, days: int, offset: int=0) -> bool:
    return now - timedelta(days = days+offset) < e.event_time <= now - timedelta(days=offset)

def to_dt(x, ref: datetime) -> datetime | None:
    if isinstance(x, datetime): d=x
    else:
        try:
            d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))

        except ValueError: return None

    if d.tzinfo is None and ref.tzinfo is not None:
        d = d.replace(tzinfo=ref.tzinfo)
    elif d.tzinfo is not None and ref.tzinfo is None:
        d = d.replace(tzinfo=None)

    return d

@dataclass
class ToolInput:
    customer_id : str
    trigger: Any
    events: list
    findings: list
    now: datetime
    extra: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable[[ToolInput], dict]
    description: str
    reads: str

TOOLS: dict[str, list[ToolSpec]] = {}

def register(*agents: str, reads: str):
    def deco(fn):
        spec = ToolSpec(fn.__name__, fn, (fn.__doc__ or "").strip(), reads)
        for a in agents:
            TOOLS.setdefault(a, []).append(spec)
        return fn
    return deco

def run_tools(agent_key: str, inp: ToolInput, ids: dict)-> dict:
    results: dict[str, Any] = {}
    for spec in TOOLS.get(agent_key, []):
        emit("tool_call", agent=agent_key, level="debug", message=spec.name,
        data={"tool": spec.name, "reads": spec.reads, "n_events_in_scope": len(inp.events)}, **ids)
        t0 = time.perf_counter()
        try:
            out = spec.fn(inp)
        except Exception as e:                       # a broken tool must not break the pipeline
            out = {"error": repr(e)}
        raw = json.dumps(out, default=str)
        if len(raw) > 3000:
            out = {"truncated": True, "preview": raw[:3000]}
        results[spec.name] = out
        emit("tool_result", agent=agent_key, message=spec.name,
             data={"tool": spec.name, "reads": spec.reads, "result": out,
                   "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}, **ids)
    return results