from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from llm.client import get_llm

DecisionVerdict = Literal["approve", "reject", "modify"]
CitationKind = Literal["event", "run", "guardrail", "audit", "trace"]

DB_PATH = Path(os.getenv("C360_DB") or Path(__file__).resolve().parents[1] / "server" / "server.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS hitl_requests (
    request_id               TEXT PRIMARY KEY,
    run_id                   TEXT NOT NULL,
    customer_id              TEXT NOT NULL,
    event_id                 TEXT NOT NULL,
    status                   TEXT NOT NULL,
    reason                   TEXT NOT NULL,
    decision_json            TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    responded_at             TEXT,
    reviewer                 TEXT,
    human_notes              TEXT,
    modified_action          TEXT,
    modified_action_subtype  TEXT,
    updated_at               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_hitl_requests_status ON hitl_requests(status, created_at);
CREATE INDEX IF NOT EXISTS ix_hitl_requests_event ON hitl_requests(event_id, created_at);

CREATE TABLE IF NOT EXISTS hitl_audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL,
    ts           TEXT NOT NULL,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    data_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_hitl_audit_request ON hitl_audit(request_id, id);
"""

ASK_TRACE_LIMIT = 30
ASK_AUDIT_LIMIT = 10
ASK_VALUE_LIMIT = 280

ASK_SYSTEM_PROMPT = """
You are a HITL decision explainer for a banking agent system.

Answer the reviewer question ONLY from the provided context.
Do not invent facts, events, IDs, timings, or policy details.
If context is insufficient, explicitly say what is missing.
Keep answer concise and operational (why it is pending, and what approve/reject/modify will do when relevant).

You must cite evidence using only allowed refs from context.
"""


class AskCitationModel(BaseModel):
    ref: str = Field(description="One allowed reference key such as EV:..., RUN:..., GR:..., AU:..., TR:...")
    why: str = Field(description="Brief reason this reference supports the answer")


class AskExplainOutput(BaseModel):
    answer: str = Field(description="Direct answer for the human reviewer")
    citations: list[AskCitationModel] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _row_to_request(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["decision"] = _loads_json(out.pop("decision_json", "{}"), {})
    return out


def _append_audit(
    conn: sqlite3.Connection,
    *,
    request_id: str,
    actor: str,
    action: str,
    data: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO hitl_audit(request_id, ts, actor, action, data_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (request_id, _now_iso(), actor, action, json.dumps(data or {}, default=str)),
    )


def _hitl_reason(decision: dict[str, Any]) -> str:
    results = decision.get("guardrail_results") or []
    reasons = []
    for g in results:
        if g.get("forced_hitl"):
            reasons.append(str(g.get("detail") or "").strip())
    if reasons:
        return "; ".join(r for r in reasons if r)[:1200]
    fallback = decision.get("notes") or "manual review required by policy"
    return str(fallback)[:1200]


def _clip_text(value: Any, limit: int = ASK_VALUE_LIMIT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep_kinds = {
        "event_received",
        "gate",
        "intervention",
        "guardrails",
        "hitl_request",
        "finalize",
        "error",
    }
    rows = [r for r in trace if str(r.get("kind")) in keep_kinds][-ASK_TRACE_LIMIT:]
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "ref": f"TR:{r.get('id')}",
                "kind": r.get("kind"),
                "agent": r.get("agent"),
                "message": _clip_text(r.get("message")),
                "data": _clip_text(r.get("data")),
            }
        )
    return out


def _summarize_audit(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = audit[-ASK_AUDIT_LIMIT:]
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "ref": f"AU:{r.get('id')}",
                "ts": r.get("ts"),
                "actor": r.get("actor"),
                "action": r.get("action"),
                "data": _clip_text(r.get("data")),
            }
        )
    return out


def _citation_kind_from_ref(ref: str) -> CitationKind:
    if ref.startswith("EV:"):
        return "event"
    if ref.startswith("RUN:"):
        return "run"
    if ref.startswith("GR:"):
        return "guardrail"
    if ref.startswith("AU:"):
        return "audit"
    return "trace"


def _deterministic_answer(
    *,
    request_id: str,
    question: str,
    req: dict[str, Any],
    decision: dict[str, Any],
    trace: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    evidence = decision.get("evidence_event_ids") or []
    guardrails = decision.get("guardrail_results") or []
    routed = [g for g in guardrails if g.get("forced_hitl") or not g.get("passed", True)]

    answer = (
        f"Request {request_id} is pending because action '{decision.get('action')}' was escalated by guardrails. "
        f"Reason: {req.get('reason')}. Evidence events: {', '.join(evidence) or 'none'}."
    )
    ql = question.lower()
    if "what changed" in ql or "modify" in ql:
        answer += " You can use /modify to override action/subtype with reviewer notes."
    elif "reject" in ql:
        answer += " Rejecting will finalize this decision as no_action (human_rejected)."
    elif "approve" in ql:
        answer += " Approving keeps the proposed action and marks it human_approved."
    elif "trace" in ql or "prompt" in ql:
        answer += f" Full trace is available and currently has {len(trace)} record(s) for this event."

    citations: list[dict[str, Any]] = [
        {"type": "event", "ref": f"EV:{req['event_id']}", "detail": "triggering event"},
        {"type": "run", "ref": f"RUN:{req['run_id']}", "detail": "processing run"},
    ]
    for g in routed:
        citations.append(
            {
                "type": "guardrail",
                "ref": f"GR:{g.get('rule_id')}",
                "detail": _clip_text(g.get("detail")),
            }
        )
    return answer, citations


def _llm_grounded_answer(
    *,
    question: str,
    req: dict[str, Any],
    decision: dict[str, Any],
    trace: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    guardrails = decision.get("guardrail_results") or []
    guardrail_refs = [
        {"ref": f"GR:{g.get('rule_id')}", "rule_id": g.get("rule_id"), "detail": _clip_text(g.get("detail"))}
        for g in guardrails
    ]
    audit_rows = _summarize_audit(req.get("audit") or [])
    trace_rows = _summarize_trace(trace)

    allowed_refs = [f"EV:{req['event_id']}", f"RUN:{req['run_id']}"]
    allowed_refs.extend([g["ref"] for g in guardrail_refs if g.get("ref")])
    allowed_refs.extend([a["ref"] for a in audit_rows if a.get("ref")])
    allowed_refs.extend([t["ref"] for t in trace_rows if t.get("ref")])
    allowed_ref_set = set(allowed_refs)

    context = {
        "question": question,
        "request": {
            "request_id": req.get("request_id"),
            "status": req.get("status"),
            "reason": req.get("reason"),
            "customer_id": req.get("customer_id"),
            "event_id": req.get("event_id"),
            "run_id": req.get("run_id"),
        },
        "decision": {
            "action": decision.get("action"),
            "action_subtype": decision.get("action_subtype"),
            "confidence_band": decision.get("confidence_band"),
            "hitl_status": decision.get("hitl_status"),
            "status": decision.get("status"),
            "evidence_event_ids": decision.get("evidence_event_ids") or [],
            "notes": _clip_text(decision.get("notes"), 700),
        },
        "guardrails": guardrail_refs,
        "audit_excerpt": audit_rows,
        "trace_excerpt": trace_rows,
        "allowed_refs": allowed_refs,
    }

    user_prompt = (
        "Reviewer question and grounded context JSON:\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "Return concise answer plus citations using only allowed refs."
    )
    llm = get_llm()
    out = llm.generate_structured(
        system_prompt=ASK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=AskExplainOutput,
    )

    citations: list[dict[str, Any]] = []
    for c in out.citations:
        if c.ref not in allowed_ref_set:
            continue
        citations.append(
            {
                "type": _citation_kind_from_ref(c.ref),
                "ref": c.ref,
                "detail": _clip_text(c.why, 220),
            }
        )

    if not citations:
        citations = [
            {"type": "event", "ref": f"EV:{req['event_id']}", "detail": "triggering event"},
            {"type": "run", "ref": f"RUN:{req['run_id']}", "detail": "processing run"},
        ]

    return out.answer.strip(), citations


def ensure_pending_hitl_request(
    *,
    run_id: str,
    customer_id: str,
    event_id: str,
    decision: dict[str, Any],
) -> dict[str, Any] | None:
    if decision.get("status") != "pending_hitl":
        return None

    conn = _connect()
    try:
        existing = conn.execute(
            """
            SELECT *
            FROM hitl_requests
            WHERE customer_id = ? AND event_id = ? AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (customer_id, event_id),
        ).fetchone()
        if existing:
            return _row_to_request(existing)

        now = _now_iso()
        request_id = "hitl_" + uuid4().hex[:12]
        reason = _hitl_reason(decision)
        conn.execute(
            """
            INSERT INTO hitl_requests(
                request_id, run_id, customer_id, event_id, status, reason, decision_json,
                created_at, responded_at, reviewer, human_notes,
                modified_action, modified_action_subtype, updated_at
            )
            VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?)
            """,
            (request_id, run_id, customer_id, event_id, reason, json.dumps(decision, default=str), now, now),
        )
        _append_audit(
            conn,
            request_id=request_id,
            actor="system",
            action="created",
            data={"reason": reason, "action": decision.get("action")},
        )
        conn.commit()
        created = conn.execute("SELECT * FROM hitl_requests WHERE request_id = ?", (request_id,)).fetchone()
        return _row_to_request(created) if created else None
    finally:
        conn.close()


def list_hitl_requests(status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM hitl_requests
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (status, max(1, min(limit, 500))),
        ).fetchall()
        return [_row_to_request(r) for r in rows]
    finally:
        conn.close()


def get_hitl_request(request_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM hitl_requests WHERE request_id = ?", (request_id,)).fetchone()
        if not row:
            return None
        req = _row_to_request(row)
        audits = conn.execute(
            "SELECT id, ts, actor, action, data_json FROM hitl_audit WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
        req["audit"] = [
            {
                "id": a["id"],
                "ts": a["ts"],
                "actor": a["actor"],
                "action": a["action"],
                "data": _loads_json(a["data_json"], {}),
            }
            for a in audits
        ]
        return req
    finally:
        conn.close()


def decide_hitl_request(
    *,
    request_id: str,
    verdict: DecisionVerdict,
    reviewer: str,
    human_notes: str | None = None,
    modified_action: str | None = None,
    modified_action_subtype: str | None = None,
) -> dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM hitl_requests WHERE request_id = ?", (request_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown request_id: {request_id}")

        current = _row_to_request(row)
        if current["status"] != "pending":
            raise ValueError(f"request {request_id} is already {current['status']}")

        decision = current["decision"]
        decision["status"] = "final"

        if verdict == "approve":
            status = "approved"
            decision["hitl_status"] = "human_approved"
        elif verdict == "reject":
            status = "rejected"
            decision["hitl_status"] = "human_rejected"
            decision["action"] = "no_action"
            decision["action_subtype"] = None
        else:
            if not modified_action:
                raise ValueError("modified_action is required when verdict=modify")
            status = "modified"
            decision["hitl_status"] = "human_modified"
            decision["action"] = modified_action
            decision["action_subtype"] = modified_action_subtype

        if human_notes:
            base = str(decision.get("notes") or "").strip()
            suffix = f"HITL note ({reviewer}): {human_notes.strip()}"
            decision["notes"] = f"{base}\n{suffix}".strip() if base else suffix

        now = _now_iso()
        conn.execute(
            """
            UPDATE hitl_requests
            SET status = ?, decision_json = ?, responded_at = ?, reviewer = ?, human_notes = ?,
                modified_action = ?, modified_action_subtype = ?, updated_at = ?
            WHERE request_id = ?
            """,
            (
                status,
                json.dumps(decision, default=str),
                now,
                reviewer,
                human_notes,
                modified_action,
                modified_action_subtype,
                now,
                request_id,
            ),
        )
        _append_audit(
            conn,
            request_id=request_id,
            actor=reviewer or "human",
            action=verdict,
            data={
                "human_notes": human_notes,
                "modified_action": modified_action,
                "modified_action_subtype": modified_action_subtype,
            },
        )
        conn.commit()
    finally:
        conn.close()

    updated = get_hitl_request(request_id)
    if not updated:
        raise KeyError(f"request {request_id} disappeared after update")
    return updated


def trace_for_hitl_request(request_id: str) -> list[dict[str, Any]]:
    req = get_hitl_request(request_id)
    if not req:
        raise KeyError(f"unknown request_id: {request_id}")

    from observe import trace_for_event

    return trace_for_event(req["event_id"], req["run_id"])


def answer_hitl_question(request_id: str, question: str) -> dict[str, Any]:
    req = get_hitl_request(request_id)
    if not req:
        raise KeyError(f"unknown request_id: {request_id}")

    q = (question or "").strip() or "why is this pending approval?"
    decision = req["decision"]
    trace = trace_for_hitl_request(request_id)
    mode = "llm"
    llm_error: str | None = None
    try:
        answer, citations = _llm_grounded_answer(
            question=q,
            req=req,
            decision=decision,
            trace=trace,
        )
    except (RuntimeError, ValueError) as e:
        mode = "fallback"
        llm_error = repr(e)
        answer, citations = _deterministic_answer(
            request_id=request_id,
            question=q,
            req=req,
            decision=decision,
            trace=trace,
        )

    conn = _connect()
    try:
        _append_audit(
            conn,
            request_id=request_id,
            actor="human",
            action="ask",
            data={"question": q, "mode": mode, "llm_error": llm_error},
        )
        conn.commit()
    finally:
        conn.close()

    return {"request_id": request_id, "question": q, "answer": answer, "citations": citations, "mode": mode}
