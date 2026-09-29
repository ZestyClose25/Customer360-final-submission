import os
from pathlib import Path

from tools.base import (INCOME_KINDS, INTERNAL_KINDS, ToolInput, dedupe, in_window, num, pfield, register, to_dt)

SRC = "KYC / profile / auth / transfer / ledger events (in-memory working memory, scoped)"
PROFILE_HINTS = ("address", "phone", "email", "mobile", "profile", "nominee", "password")
AUTH_HINTS = ("login", "auth", "signin", "sign_in")
LARGE = 10_000          # reporting-style threshold
NEAR = 9_000            # "just under" band used for structuring detection
DOMESTIC = {"US", "USA", "", "NONE"}


def _load_watchlist() -> set[str]:
    """The stream contains no sanctions list. Drop one name per line in src/data/watchlist.txt
    (or set C360_WATCHLIST=/path/file.txt). Empty by default -> the tool reports list_size 0."""
    p = Path(os.getenv("C360_WATCHLIST") or Path(__file__).resolve().parents[1] / "data" / "watchlist.txt")
    try:
        return {ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")}
    except OSError:
        return set()


WATCHLIST = _load_watchlist()


def _kind(e) -> str:
    return str(pfield(e.payload, "txn_kind", "")).lower()


def _movements(inp: ToolInput):
    """External money movements only: (event, amount, 'in'|'out'). Own-account transfers and real
    income are excluded; a ledger ach_withdrawal that mirrors an ach_wire event is counted once."""
    evs = dedupe(inp.events)
    wires = [(e, num(pfield(e.payload, "amount"))) for e in evs if e.event_type == "outbound_transfer"]
    out = []
    for e in evs:
        a, k = num(pfield(e.payload, "amount")), _kind(e)
        if a is None or k in INTERNAL_KINDS:
            continue
        a = abs(a)
        if e.event_type == "outbound_transfer":
            out.append((e, a, "out"))
        elif e.event_type == "withdrawal":
            if any(w is not None and abs(w) == a and abs((we.event_time - e.event_time).days) <= 1
                   for we, w in wires):
                continue
            out.append((e, a, "out"))
        elif e.event_type == "deposit" and k not in INCOME_KINDS:
            out.append((e, a, "in"))
    return out


@register("compliance", reads=SRC)
def large_transfer_check(inp: ToolInput) -> dict:
    """External movements in 30d: large (>=10k), just-under-threshold clusters, foreign counterparties, lump-sum inflows."""
    mv = [m for m in _movements(inp) if in_window(m[0], inp.now, 30)]
    large = [m for m in mv if m[1] >= LARGE]
    near7 = [m for m in mv if NEAR <= m[1] < LARGE and in_window(m[0], inp.now, 7)]
    foreign = [m for m in mv if str(pfield(m[0].payload, "country", "")).upper() not in DOMESTIC]
    return {
        "n_external_movements_30d": len(mv),
        "large": [{"event_id": e.event_id, "amount": a, "direction": d,
                   "counterparty": pfield(e.payload, "merchant"), "country": pfield(e.payload, "country"),
                   "status": pfield(e.payload, "status")} for e, a, d in large][:5],
        "structuring_suspected": len(near7) >= 2,
        "foreign_counterparty_events": [e.event_id for e, _, _ in foreign][:5],
        "lump_sum_inflows_30d": [{"event_id": e.event_id, "amount": a} for e, a, d in mv if d == "in" and a >= LARGE][:3],
    }


@register("compliance", reads=SRC)
def profile_changes(inp: ToolInput) -> dict:
    """Address/phone/email/etc. changes in 14d; several distinct kinds in 7d = takeover pattern."""
    ch = [e for e in dedupe(inp.events) if any(h in e.event_type.lower() for h in PROFILE_HINTS)
          and in_window(e, inp.now, 14)]
    return {"changes_14d": len(ch), "types": sorted({e.event_type for e in ch}),
            "multi_change_7d": len({e.event_type for e in ch if in_window(e, inp.now, 7)}) >= 2}


@register("compliance", reads=SRC)
def kyc_status(inp: ToolInput) -> dict:
    """Latest KYC event: status and whether the document has expired."""
    k = sorted([e for e in dedupe(inp.events) if "kyc" in e.event_type.lower()], key=lambda e: e.event_time)
    if not k:
        return {"has_kyc_events": False}
    last = k[-1]
    exp = to_dt(pfield(last.payload, "expiry"), inp.now) if pfield(last.payload, "expiry") else None
    return {"has_kyc_events": True, "last_kyc_event_type": last.event_type,
            "status": pfield(last.payload, "status"),
            "expired": bool(exp and exp < inp.now),
            "expires_in_days": (exp - inp.now).days if exp else None}


@register("compliance", reads=SRC)
def auth_anomalies(inp: ToolInput) -> dict:
    """Failed logins and devices seen in the last 24h; flags a device never seen before."""
    auth = [e for e in dedupe(inp.events) if any(h in e.event_type.lower() for h in AUTH_HINTS)]
    day = [e for e in auth if in_window(e, inp.now, 1)]
    failed = [e for e in day if "fail" in e.event_type.lower()
              or str(pfield(e.payload, "status", "")).lower() in ("failed", "failure", "denied")]
    old_dev = {str(pfield(e.payload, "device")) for e in auth if not in_window(e, inp.now, 1)
               and pfield(e.payload, "device")}
    new_dev = {str(pfield(e.payload, "device")) for e in day if pfield(e.payload, "device")} - old_dev
    return {"failed_24h": len(failed), "new_devices_24h": sorted(new_dev),
            "new_device": bool(new_dev and old_dev)}


@register("compliance", reads="src/data/watchlist.txt (loaded at startup; empty unless you add names)")
def watchlist_check(inp: ToolInput) -> dict:
    """Does the counterparty/merchant of the triggering event match a watchlist name?"""
    name = str(pfield(inp.trigger.payload, "merchant", "")).lower() if inp.trigger else ""
    return {"applicable": bool(name), "list_size": len(WATCHLIST),
            "matches": sorted(w for w in WATCHLIST if name and w in name)}