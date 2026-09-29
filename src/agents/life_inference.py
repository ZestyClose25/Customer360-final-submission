import json
import time
from typing import List, Literal

from pydantic import BaseModel, Field

from agents.critique import BANDS, band_str
from llm.client import get_llm
from observe import emit

from tools import ToolInput, run_tools, replay_now

LIFE_EVENTS = [
    "job_loss_or_income_drop", "new_job_or_income_rise", "marriage_or_partnership",
    "new_child", "home_purchase_or_move", "education_or_tuition", "retirement",
    "business_started_or_growth", "medical_or_family_emergency", "debt_stress",
    "wealth_or_investment_opportunity", "travel_or_relocation_abroad",
    "possible_fraud_or_account_takeover", "churn_risk", "other", "no_clear_life_event",
]
LifeEvent = Literal[tuple(LIFE_EVENTS)]


class HypothesisDraft(BaseModel):
    life_event: LifeEvent
    confidence_band: Literal["low", "medium", "high"]
    rationale: str = Field(description="1-3 sentences tying the findings together, no new facts")
    supporting_finding_ids: List[str] = Field(description="ids like F0, F2 from the findings list")
    contradicting_finding_ids: List[str] = Field(default_factory=list)


class LifeInferenceOutput(BaseModel):
    summary: str = Field(description="one line: what is probably going on with this customer")
    hypotheses: List[HypothesisDraft] = Field(default_factory=list, description="most likely first, max 3")
    urgency: Literal["none", "low", "medium", "high"]
    risk_themes: List[str] = Field(default_factory=list, description="short snake_case, e.g. churn, fraud, financial_distress")
    opportunity_themes: List[str] = Field(default_factory=list, description="short snake_case, e.g. loan_offer, goal_based_savings")
    open_questions: List[str] = Field(default_factory=list, description="what would confirm/refute the top hypothesis")


LIFE_SYSTEM_PROMPT = f"""
You are the Life-Inference Agent in an Agentic Customer 360 system.

You receive findings from specialist agents that a critique agent has ALREADY approved.
Infer the most probable life event(s) behind them.

Rules:
1. Use ONLY the given findings. Never invent facts, amounts, dates or events.
2. Every hypothesis must list supporting_finding_ids (F0, F1...). No support -> do not output it.
3. Give at most 3 hypotheses, most likely first. Include a competing explanation when one is
   genuinely plausible (e.g. income drop could be job loss OR just a salary-date shift).
4. Confidence: high only when several independent findings agree; medium for a coherent but
   thinly evidenced picture; low for weak signals. Do not be dramatic.
5. If the findings do not add up to a life event, return the single hypothesis
   "no_clear_life_event" and urgency "none".
6. Do NOT recommend a final action or write customer messages. Only give risk_themes,
   opportunity_themes and open_questions; another agent designs the intervention.
7. Urgency: high only for fraud/takeover signals or acute financial distress.

Allowed life_event labels: {LIFE_EVENTS}
"""


def _finding_view(idx: int, f) -> dict:
    return {"id": f"F{idx}", "agent": f.agent_name, "type": f.finding_type, "finding": f.finding,
            "confidence": band_str(f.confidence_band), "evidence": f.evidence_event_ids}


def _apply_guards(draft: LifeInferenceOutput, findings: list) -> list[dict]:
    """Validate hypotheses against the real findings. Returns clean dicts."""
    by_id = {f"F{i}": f for i, f in enumerate(findings)}
    clean = []
    for h in draft.hypotheses[:3]:
        sup = [i for i in dict.fromkeys(h.supporting_finding_ids) if i in by_id]
        if h.life_event != "no_clear_life_event" and not sup:
            continue
        band = h.confidence_band
        if sup:
            best = max(BANDS.index(band_str(by_id[i].confidence_band)) for i in sup)
            agents = {by_id[i].agent_name for i in sup}
            ceiling = min(best + (1 if len(agents) >= 2 else 0), len(BANDS) - 1)
            if BANDS.index(band) > ceiling:
                band = BANDS[ceiling]
        clean.append({
            "life_event": h.life_event, "confidence_band": band, "rationale": h.rationale,
            "supporting_finding_ids": sup,
            "supporting_agents": sorted({by_id[i].agent_name for i in sup}),
            "evidence_event_ids": sorted({e for i in sup for e in by_id[i].evidence_event_ids}),
            "contradicting_finding_ids": [i for i in h.contradicting_finding_ids if i in by_id],
        })
    return clean


async def run_life_inference(state, llm=None) -> dict:
    """LangGraph node body. Returns a PARTIAL state update."""
    llm = llm or get_llm()
    wm = state["working_memory"]
    event = state["event"]
    ids = dict(customer_id=wm.customer_id, event_id=event.event_id)
    findings = list(state.get("reviewed_findings") or [])
    critique = state.get("critique") or {}
    t0 = time.perf_counter()

    emit("agent_start", agent="life_inference", message=f"inferring from {len(findings)} approved finding(s)",
         data={"n_context_events": 0, "n_prior_findings": len(wm.agent_findings)}, **ids)

    none_result = {"summary": "no approved findings -> no life event", "hypotheses": [
        {"life_event": "no_clear_life_event", "confidence_band": "low", "rationale": "no approved findings",
         "supporting_finding_ids": [], "supporting_agents": [], "evidence_event_ids": [],
         "contradicting_finding_ids": []}],
        "urgency": "none", "risk_themes": [], "opportunity_themes": [], "open_questions": []}

    if not findings:
        emit("life_inference", agent="life_inference", message=none_result["summary"], data=none_result, **ids)
        return {"life_inference": none_result}

    tool_results = run_tools(
        "life_inference",
        ToolInput(customer_id=wm.customer_id, trigger=None, events=wm.recent_events, findings=findings, now=replay_now(wm.recent_events, event.event_time), extra={"life_history": state.get("life_history") or []}),
        ids
    )

    history = [{"agent": f.agent_name, "type": f.finding_type, "finding": f.finding[:200],
                "confidence": band_str(f.confidence_band)} for f in wm.agent_findings[-8:]]
    user_prompt = f"""
Customer: {wm.customer_id}
Triggering event: {event.event_id} ({event.source_system}/{event.event_type})

APPROVED FINDINGS (this event):
{json.dumps([_finding_view(i, f) for i, f in enumerate(findings)], indent=1)}

CRITIQUE NOTES: {critique.get("summary", "")}
Contradictions flagged: {json.dumps(critique.get("contradictions", []))}

TOOLS RESULTS (deterministic; includes life phases already inferred for this customer earlier -- 
weigh them: a phase inferred weeks ago persist unless the new findings contradict it):
{json.dumps(tool_results, indent=1, default=str)}

ACCEPTED HISTORY (older findings, context only -- cannot be cited as support):
{json.dumps(history, indent=1)}

Infer the probable life event(s).
"""
    emit("llm_prompt", agent="life_inference", level="debug",
         message=f"inference prompt ({len(user_prompt)} chars, {len(findings)} findings)",
         data={"system": LIFE_SYSTEM_PROMPT, "user": user_prompt}, **ids)
    try:
        out = await llm.agenerate_structured(
            system_prompt=LIFE_SYSTEM_PROMPT, user_prompt=user_prompt, response_schema=LifeInferenceOutput)
    except Exception as e:
        print(f"   ! life_inference agent ERROR: {e!r}")
        emit("agent_error", agent="life_inference", message=repr(e), **ids)
        return {"life_inference": {**none_result, "summary": f"inference failed: {e!r}"[:200]}}
    emit("llm_response", agent="life_inference", level="debug",
         message=f"inference response [{time.perf_counter() - t0:.1f}s]",
         data={"response": out.model_dump(mode="json"), "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)

    hyps = _apply_guards(out, findings)
    if not hyps:
        hyps = none_result["hypotheses"]
    result = {
        "summary": out.summary, "hypotheses": hyps,
        "urgency": out.urgency if hyps[0]["life_event"] != "no_clear_life_event" else "none",
        "risk_themes": out.risk_themes, "opportunity_themes": out.opportunity_themes,
        "open_questions": out.open_questions,
    }
    top = hyps[0]
    print(f"   → life_inference: {top['life_event']} ({top['confidence_band']}), urgency={result['urgency']} "
          f"[{time.perf_counter() - t0:.1f}s]")
    emit("life_inference", agent="life_inference", message=out.summary, data=result, **ids)
    upd: dict = {"life_inference": result}
    if top["life_event"] != "no_clear_life_event":
        upd["life_history"] = (state.get("life_history") or [])[-9:] + [{
            "at": event.event_time.isoformat(), "event_id": event.event_id,
            "life_event": top["life_event"], "confidence_band": top["confidence_band"],
            "urgency": result["urgency"]}]
    emit("agent_done", agent="life_inference", message=out.summary,
         data={"n_findings": len(hyps), "n_handoffs": 0, "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)
    return upd
