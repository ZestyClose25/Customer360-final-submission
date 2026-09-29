import json
import time
from datetime import datetime, timezone
from typing import List, Literal

from pydantic import BaseModel, Field

from llm.client import get_llm
from observe import emit

from tools import ToolInput, run_tools, replay_now

MAX_EVIDENCE_EVENTS = 20
MAX_CRITIQUE_ROUNDS = 2      # 1st pass may request specialists; 2nd pass reviews their output
BANDS = ["low", "medium", "high"]

AgentName = Literal["transaction", "support", "investment", "compliance", "engagement"]


def band_str(b) -> str:
    """confidence_band may be an Enum or a str depending on memory.schema."""
    return str(getattr(b, "value", b)).lower()


# ---------------- LLM output schema ----------------
class FindingVerdict(BaseModel):
    finding_index: int = Field(description="index of the finding as listed in the prompt (F0, F1... -> 0, 1...)")
    verdict: Literal["keep", "downgrade", "drop"]
    new_confidence: Literal["low", "medium", "high"] | None = Field(
        default=None, description="only when verdict=downgrade")
    reason: str = Field(description="one short sentence")


class MissingSpecialist(BaseModel):
    agent: AgentName
    reason: str = Field(description="what that specialist's data would confirm or refute")


class CritiqueOutput(BaseModel):
    summary: str = Field(description="one line: overall quality of the findings")
    verdicts: List[FindingVerdict] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list,
                                      description="findings that conflict with each other or with earlier findings")
    missing_specialists: List[MissingSpecialist] = Field(default_factory=list)


CRITIQUE_SYSTEM_PROMPT = """
You are the Critique Agent in an Agentic Customer 360 system.

Specialist agents (transaction, engagement, support, investment, compliance) have
just produced findings about a customer. You review them. You do NOT produce new
findings and you do NOT infer the customer's life event.

For EACH finding decide:
- keep: the cited evidence events clearly support the claim.
- downgrade: plausible but the evidence is thin, single-event, or the wording
  overstates it. Give a lower new_confidence.
- drop: the claim is not supported by the cited events, is speculation, or is a
  duplicate of another finding.

Also:
- List contradictions between findings (or with earlier findings).
- Request a specialist that has NOT yet run ONLY if its data source is needed to
  confirm or refute an important finding. Never request one that already ran.
  Prefer requesting nobody; an unnecessary request wastes a round.

Judge only from the events and findings shown. Be strict but not pedantic:
routine, well-evidenced findings should simply be kept.
"""


def deterministic_checks(findings, valid_ids: set[str]) -> dict[int, dict]:
    """Cheap pre-checks. Returns {finding_index: {"verdict","reason"}} for hard failures."""
    out: dict[int, dict] = {}
    seen: dict[tuple, int] = {}
    for i, f in enumerate(findings):
        ev = [e for e in f.evidence_event_ids if e in valid_ids]
        if not ev:
            out[i] = {"verdict": "drop", "reason": "no valid evidence event ids"}
            continue
        key = (f.agent_name, f.finding_type)
        if key in seen:
            out[i] = {"verdict": "drop", "reason": f"duplicate of F{seen[key]}"}
        else:
            seen[key] = i
    return out


async def run_critique(state, llm=None) -> dict:
    llm = llm or get_llm()
    wm = state["working_memory"]
    event = state["event"]
    completed = list(dict.fromkeys(state.get("completed_agents") or []))
    rnd = state.get("critique_round", 0) + 1
    prev = state.get("critique") or {}
    prev_reviewed_agents = set(prev.get("reviewed_agents") or [])
    prev_kept = list(state.get("reviewed_findings") or [])
    prev_detail = list(prev.get("verdicts") or [])
    # round 2 only reviews findings from specialists the first pass didn't see
    findings = [f for f in (state.get("agent_findings") or []) if f.agent_name not in prev_reviewed_agents]
    ids = dict(customer_id=wm.customer_id, event_id=event.event_id)
    t0 = time.perf_counter()

    emit("agent_start", agent="critique",
         message=f"reviewing {len(findings)} finding(s)",
         data={"n_context_events": len(wm.recent_events), "sources": [],
               "n_prior_findings": len(wm.agent_findings)}, **ids)

    if not findings:
        emit("critique_verdict", agent="critique", message="nothing to review", data={"verdicts": []}, **ids)
        return {"critique": {**prev, "summary": prev.get("summary", "nothing to review"),
                             "verdicts": prev_detail, "reviewed_agents": sorted(prev_reviewed_agents)},
                "critique_round": rnd, "critique_requests": None}

    valid_ids = {e.event_id for e in wm.recent_events}
    hard = deterministic_checks(findings, valid_ids)

    # evidence events the findings cite (compact) so the LLM can actually check them
    cited = {i for f in findings for i in f.evidence_event_ids if i in valid_ids}
    ev = [e.model_dump(mode="json") for e in wm.recent_events if e.event_id in cited][-MAX_EVIDENCE_EVENTS:]
    listed = [
        {"index": i, "agent": f.agent_name, "type": f.finding_type, "finding": f.finding,
         "confidence": band_str(f.confidence_band), "evidence": f.evidence_event_ids,
         **({"pre_check": hard[i]["reason"]} if i in hard else {})}
        for i, f in enumerate(findings)
    ]
    earlier = [
        {"agent": f.agent_name, "type": f.finding_type, "finding": f.finding[:200],
         "confidence": band_str(f.confidence_band)}
        for f in wm.agent_findings[-6:]
    ]
    not_run = [a for a in ["transaction", "support", "investment", "compliance", "engagement"]
               if a not in completed]

    tool_results = run_tools(
        "critique",
        ToolInput(customer_id=wm.customer_id, trigger=None, events=wm.recent_events, findings=findings, now=replay_now(wm.recent_events, event.event_time)),
        ids
    )

    user_prompt = f"""
Customer: {wm.customer_id}
Triggering event: {event.event_id} ({event.source_system}/{event.event_type})

DETERMINISTIC CHECKS (tool results; trust these over your own reading):
{json.dumps(tool_results, indent=1, default=str)}

FINDINGS TO REVIEW (this event):
{json.dumps(listed, indent=1)}

EVIDENCE EVENTS CITED BY THE FINDINGS:
{json.dumps(ev, indent=1, default=str)}

EARLIER FINDINGS (context, already accepted):
{json.dumps(earlier, indent=1)}

Specialists that already ran this event: {completed}
Specialists you may still request: {not_run}
"""
    emit("llm_prompt", agent="critique", level="debug",
         message=f"critique prompt ({len(user_prompt)} chars, {len(findings)} findings)",
         data={"system": CRITIQUE_SYSTEM_PROMPT, "user": user_prompt}, **ids)
    try:
        out = await llm.agenerate_structured(
            system_prompt=CRITIQUE_SYSTEM_PROMPT, user_prompt=user_prompt, response_schema=CritiqueOutput)
    except Exception as e:
        # fail-safe: on LLM failure keep everything except the hard-check failures
        print(f"   ! critique agent ERROR: {e!r}")
        emit("agent_error", agent="critique", message=repr(e), **ids)
        out = CritiqueOutput(summary="critique LLM failed; only deterministic checks applied")
    emit("llm_response", agent="critique", level="debug",
         message=f"critique response [{time.perf_counter() - t0:.1f}s]",
         data={"response": out.model_dump(mode="json"), "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)

    # ---- merge: deterministic failures always win over the LLM ----
    verdicts: dict[int, dict] = {}
    for v in out.verdicts:
        if 0 <= v.finding_index < len(findings):
            verdicts[v.finding_index] = {"verdict": v.verdict, "new_confidence": v.new_confidence, "reason": v.reason}
    for i, h in hard.items():
        verdicts[i] = {"verdict": h["verdict"], "new_confidence": None, "reason": h["reason"]}
    for i in range(len(findings)):
        verdicts.setdefault(i, {"verdict": "keep", "new_confidence": None, "reason": "not flagged"})

    now = datetime.now(timezone.utc)
    kept, dropped = [], []
    for i, f in enumerate(findings):
        v = verdicts[i]
        if v["verdict"] == "drop":
            dropped.append(f)
            continue
        if v["verdict"] == "downgrade" and v["new_confidence"]:
            # a critique can only LOWER confidence, never raise it
            if BANDS.index(v["new_confidence"]) < BANDS.index(band_str(f.confidence_band)):
                f = f.model_copy(update={"confidence_band": v["new_confidence"], "updated_at": now})
        kept.append(f)

    # requests -> manager (only specialists that haven't run; manager re-validates)
    requests = [{"to_agent": m.agent, "reason": m.reason}
                for m in out.missing_specialists if m.agent not in completed]
    if rnd >= MAX_CRITIQUE_ROUNDS:      # last pass: never ask for more work
        requests = []

    detail = [{"finding_index": i, "agent": findings[i].agent_name, "type": findings[i].finding_type, **verdicts[i]}
              for i in range(len(findings))]
    emit("critique_verdict", agent="critique", message=out.summary,
         data={"verdicts": detail, "contradictions": out.contradictions, "requests": requests,
               "kept": len(kept), "dropped": len(dropped), "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)
    print(f"   → critique: kept {len(kept)}, dropped {len(dropped)}, requests {[r['to_agent'] for r in requests]} "
          f"[{time.perf_counter() - t0:.1f}s]")
    emit("agent_done", agent="critique", message=out.summary,
         data={"n_findings": len(kept), "n_handoffs": len(requests),
               "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)

    return {
        "reviewed_findings": prev_kept + kept,          # what life_inference is allowed to use
        "critique": {"summary": out.summary, "verdicts": prev_detail + detail,
                     "contradictions": list(prev.get("contradictions") or []) + out.contradictions,
                     "reviewed_agents": sorted(prev_reviewed_agents | {f.agent_name for f in findings})},
        "critique_requests": requests or None,
        "critique_round": rnd,
    }
