import json
import time
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from llm.client import get_llm
from memory.schema import CurrentDecision
from observe import emit

LIFE_TO_STATE = {
    "job_loss_or_income_drop": "job_loss_or_income_disruption",
    "new_job_or_income_rise": "job_change_or_promotion",
    "marriage_or_partnership": "marriage_or_relationship_change",
    "new_child": "new_child_life_event",
    "home_purchase_or_move": "relocation",
    "education_or_tuition": "no_significant_event",
    "retirement": "retirement_transition",
    "business_started_or_growth": "small_business_cashflow_event",
    "medical_or_family_emergency": "medical_hardship",
    "debt_stress": "financial_distress_general",
    "wealth_or_investment_opportunity": "wealth_growth_or_windfall",
    "travel_or_relocation_abroad": "relocation",
    "possible_fraud_or_account_takeover": "potential_fraud_or_takeover",
    "churn_risk": "churn_risk",
    "other": "no_significant_event",
    "no_clear_life_event": "no_significant_event",
}

ALLOWED_ACTIONS = {
    "no_significant_event": ["no_action"],
    "new_child_life_event": ["personalized_offer", "proactive_retention_outreach", "no_action"],
    "marriage_or_relationship_change": ["personalized_offer", "relationship_manager_escalation", "no_action"],
    "job_change_or_promotion": ["personalized_offer", "relationship_manager_escalation", "no_action"],
    "job_loss_or_income_disruption": ["support_intervention", "relationship_manager_escalation", "no_action"],
    "medical_hardship": ["support_intervention", "relationship_manager_escalation", "no_action"],
    "financial_distress_general": ["support_intervention", "relationship_manager_escalation", "no_action"],
    "relocation": ["personalized_offer", "proactive_retention_outreach", "no_action"],
    "retirement_transition": ["relationship_manager_escalation", "personalized_offer", "no_action"],
    "wealth_growth_or_windfall": ["relationship_manager_escalation", "personalized_offer", "no_action"],
    "potential_fraud_or_takeover": ["compliance_fraud_hold", "relationship_manager_escalation", "no_action"],
    "elder_vulnerability_or_scam_risk": ["compliance_fraud_hold", "relationship_manager_escalation",
                                         "support_intervention", "no_action"],
    "churn_risk": ["proactive_retention_outreach", "personalized_offer", "no_action"],
    "small_business_cashflow_event": ["personalized_offer", "relationship_manager_escalation", "no_action"],
}

ACT_MIN_BAND = "high"
ACT_ON_URGENCY = "high"

class InterventionOutput(BaseModel):
    action: Literal[
        "no_action", "proactive_retention_outreach", "relationship_manager_escalation",
        "personalized_offer", "support_intervention", "compliance_fraud_hold",
    ]
    action_subtype: Optional[str] = Field(
        default=None, description="short snake_case, e.g. medical_hardship_payment_plan")
    notes: str = Field(description="1-2 sentences: why this action, why now, grounded in the findings")

INTERVENTION_SYSTEM_PROMPT = """
You are the Intervention Agent in an Agentic Customer 360 system.

A life-inference agent has already decided WHAT is happening to the customer. You decide the
single best bank action, choosing ONLY from the allowed actions you are given.

Rules:
1. Use only the facts provided. Never invent amounts, dates or events.
2. Prefer no_action when the situation is ambiguous or a red-herring explanation is plausible.
3. Fraud/takeover hold only when the evidence points to it; never for a one-off large transfer.
4. action_subtype: short snake_case describing the concrete offer/help (or null).
5. notes: 1-2 sentences a human reviewer can read to understand the decision.
"""

def _decision(state_label, band, action, subtype, evidence, notes) -> CurrentDecision:
    now = datetime.now(timezone.utc)
    return CurrentDecision(
        inferred_state=state_label, confidence_band=band, action=action, action_subtype=subtype,
        evidence_event_ids=evidence, notes=notes, status="draft", created_at=now, updated_at=now
    )

async def run_intervention(state, llm=None) -> dict:
    llm = llm or get_llm()
    wm, event = state["working_memory"], state["event"]
    ids = dict(customer_id=wm.customer_id, event_id=event.event_id)
    li = state.get("life_inference") or {}
    hyps = li.get("hypotheses") or []
    t0 = time.perf_counter()

    emit("agent_start", agent="intervention", message="choosing action", data={}, **ids)
    top = hyps[0] if hyps else {"life_event":"no_clear_life_event", "confidence_band":"low", "rationale":"", "evidence_event_ids":[]}
    inferred = LIFE_TO_STATE.get(top["life_event"], "no_significant_event")
    band = top["confidence_band"]
    urgency = li.get("urgency", "none")
    evidence = top.get("evidence_event_ids", [])
    allowed = ALLOWED_ACTIONS[inferred]

    bands = ["low", "medium", "high"]
    ready = bands.index(band) >= bands.index(ACT_MIN_BAND) or urgency == ACT_ON_URGENCY
    if inferred == "no_significant_event" or not ready:
        why = ("no significant event inferred" if inferred == "no_significant_event" else f"confidence {band}/urgency {urgency}: monitor, not act yet")
        d = _decision(inferred, band, "no_action", None, evidence, f"{why}. {top['rationale']}".strip())
        emit("intervention", agent="intervention", message=f"no_action ({why})", data=d.model_dump(mode="json"), **ids)
        return {"proposed_decision": d}

    findings = state.get("reviewed_findings") or []
    user_prompt = f"""
    Customer: {wm.customer_id}
    Inferred_state: {inferred} (confidence {band}, urgency {urgency})
    Rationale: {top["rationale"]}
    Open questions: {json.dumps(li.get("open_questions", []))}
    Risk themes: {json.dumps(li.get('risk_themes', []))}
    Opportunity themes: {json.dumps(li.get('opportunity_themes', []))}

    Supporting findings: {json.dumps([{'agent': f.agent_name, 'type': f.finding_type, 'finding': f.finding} for f in findings], indent=1)}

    Previous decision for this customer: {wm.current_decision.model_dump(mode='json', include={'inferred_state', 'action', 'action_subtype'}) if wm.current_decision else None}

    ALLOWED ACTIONS: {allowed}
    """
    emit("llm_prompt", agent="intervention", level="debug", 
        message=f"intervention prompt ({len(user_prompt)} chars)",
        data={"system": INTERVENTION_SYSTEM_PROMPT, "user": user_prompt}, **ids)

    try:
        out = await llm.agenerate_structured(
            system_prompt=INTERVENTION_SYSTEM_PROMPT, user_prompt=user_prompt,
            response_schema=InterventionOutput
        )
    except Exception as e:
        emit("agent_error", agent="intervention", message=repr(e), **ids)
        d = _decision(inferred, band, "no_action", None, evidence, f"intervention LLM failed: {e!r}"[:200])
        return {"proposed_decision": d}
    emit("llm_response", agent="intervention", level="debug",
         message=f"intervention response [{time.perf_counter() - t0:.1f}s]",
         data={"response": out.model_dump(mode="json")}, **ids)

    action, subtype, notes = out.action, out.action_subtype, out.notes
    if action not in allowed:      # hard rule: LLM can't pick an illegal action for this state
        notes = f"[{action} not allowed for {inferred}; downgraded] {notes}"
        action, subtype = "no_action", None

    d = _decision(inferred, band, action, subtype, evidence, notes)
    print(f"   → intervention: {inferred} ({band}) -> {action} [{time.perf_counter() - t0:.1f}s]")
    emit("intervention", agent="intervention", message=f"{action}: {notes[:120]}",
         data=d.model_dump(mode="json"), **ids)
    emit("agent_done", agent="intervention", message=action,
         data={"llm": True, "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)
    return {"proposed_decision": d}