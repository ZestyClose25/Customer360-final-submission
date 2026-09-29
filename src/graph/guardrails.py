from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.intervention import ALLOWED_ACTIONS
from memory.schema import CurrentDecision, GuardrailResult

BANDS = ["low", "medium", "high"]

MIN_BAND = {"compliance_fraud_hold": "medium"}
URGENT_MIN_BAND = "medium"
MIN_EVIDENCE = 2
DISTRESS_LIFE_EVENTS = {"job_loss_or_income_drop", "medical_or_family_emergency", "debt_stress"}
DISTRESS_STATES = {"job_loss_or_income_disruption", "medical_hardship", "financial_distress_general"}
DISTRESS_WINDOW_DAYS = 30
HITL_ALWAYS = {"compliance_fraud_hold", "relationship_manager_escalation", "support_intervention", "personalized_offer"}

def _recent_distress(life_history:list, ref: datetime) -> bool:
    for h in life_history:
        if h.get("life_event") not in DISTRESS_LIFE_EVENTS:
            continue
        try:
            at = datetime.fromisoformat(str(h["at"]).replace("Z", "+00:00"))
            if at.tzinfo is None and ref.tzinfo is not None:
                at = at.replace(tzinfo=ref.tzinfo)
            if timedelta(0) <= ref - at <= timedelta(days=DISTRESS_WINDOW_DAYS):
                return True
        except Exception:
            continue
    return False

def hard_stop_decision(event, gate: dict) -> CurrentDecision:
    fraud = any("FRAUD" in r for r in gate.get("rule_ids", []))
    now = datetime.now(timezone.utc)
    return CurrentDecision(
        inferred_state="potential_fraud_or_takeover" if fraud else "churn_risk",
        confidence_band="medium",
        action="compliance_fraud_hold" if fraud else "relationship_manager_escalation",
        action_subtype="hard_stop_fraud_keyword" if fraud else "legal_threat_review",
        evidence_event_ids=[event.event_id],
        notes=f"deterministic hard stop: {(gate.get('reasons') or [''])[0]}",
        status="draft", created_at=now, updated_at=now,    
    )

def apply_guardrails(decision: CurrentDecision, *, memory, critique: dict, life_inference: dict, life_history: list, gate: dict, ref_time: datetime) -> CurrentDecision:
    d = decision.model_copy(deep=True)
    results: list[GuardrailResult] = []
    by_id = {e.event_id: e for e in memory.recent_events}
    hard_stop = gate.get("route") == "hard_stop"
    urgency = (life_inference or {}).get("urgency", "none")

    def block(rule_id: str, detail: str):
        results.append(GuardrailResult(rule_id=rule_id, passed=False, modified_field="action", detail=f"{detail}; {d.action} -> no_action"))
        d.notes = f"[blocked by {rule_id}: {detail}] {d.notes}"
        d.action, d.action_subtype = "no_action", None

    def ok(rule_id: str, detail: str):
        results.append(GuardrailResult(rule_id=rule_id, passed=True, detail=detail))

    # ---- action gates (skipped for hard stops: those are policy, they go to a human) ----
    if d.action != "no_action" and not hard_stop:
        # G1: action must be legal for the inferred state
        if d.action not in ALLOWED_ACTIONS.get(d.inferred_state, ["no_action"]):
            block("ACTION_ALLOWED_FOR_STATE", f"{d.action} not allowed for {d.inferred_state}")
        else:
            ok("ACTION_ALLOWED_FOR_STATE", f"{d.action} is legal for {d.inferred_state}")

    if d.action != "no_action" and not hard_stop:
        # G2: evidence ids must be real; enough of them
        valid = [i for i in d.evidence_event_ids if i in by_id]
        if len(valid) != len(d.evidence_event_ids):
            results.append(GuardrailResult(rule_id="EVIDENCE_VALID", passed=False,
                                           modified_field="evidence_event_ids",
                                           detail=f"dropped {len(d.evidence_event_ids) - len(valid)} unknown evidence id(s)"))
            d.evidence_event_ids = valid
        if len(valid) < MIN_EVIDENCE:
            block("EVIDENCE_MIN", f"{len(valid)} valid evidence event(s) < {MIN_EVIDENCE}")
        else:
            ok("EVIDENCE_MIN", f"{len(valid)} evidence events")

    if d.action != "no_action" and not hard_stop:
        # G3: confidence floor (urgent cases may act one band lower)
        floor = URGENT_MIN_BAND if urgency == "high" else MIN_BAND.get(d.action, "high")
        if BANDS.index(d.confidence_band) < BANDS.index(floor):
            block("CONFIDENCE_FLOOR", f"confidence {d.confidence_band} < required {floor} for {d.action}")
        else:
            ok("CONFIDENCE_FLOOR", f"{d.confidence_band} >= {floor}")

    if d.action == "compliance_fraud_hold" and not hard_stop:
        # G4: a fraud hold needs corroboration from >= 2 distinct kinds of event
        kinds = {(by_id[i].source_system, by_id[i].event_type) for i in d.evidence_event_ids if i in by_id}
        if len(kinds) < 2:
            block("FRAUD_CORROBORATION", f"evidence spans {len(kinds)} event kind(s); need >= 2")
        else:
            ok("FRAUD_CORROBORATION", f"{len(kinds)} distinct event kinds")

    if d.action == "personalized_offer":
        # G5: never sell to a customer who is (recently) in distress
        prev = memory.current_decision
        if _recent_distress(life_history, ref_time) or (prev and prev.inferred_state in DISTRESS_STATES):
            block("NO_OFFER_WHILE_DISTRESSED", "recent hardship signal on this customer")
        else:
            ok("NO_OFFER_WHILE_DISTRESSED", "no recent distress")

    # ---- G6: same decision already issued -> don't re-issue / re-open HITL ----
    prev = memory.current_decision
    if (d.action != "no_action" and prev and prev.status in ("pending_hitl", "final")
            and (prev.inferred_state, prev.action) == (d.inferred_state, d.action)):
        d.action_subtype = prev.action_subtype or d.action_subtype
        d.hitl_status, d.status, d.created_at = prev.hitl_status, prev.status, prev.created_at
        ok("NO_REPEAT", "same state+action already issued; no new HITL request")
        d.guardrail_results, d.updated_at = results, datetime.now(timezone.utc)
        return d

    # ---- G7: HITL routing ----
    if d.action == "no_action":
        d.hitl_status, d.status = "auto_approved", "final"
        ok("HITL_ROUTE", "no action; nothing to review")
    else:
        why = []
        if d.action in HITL_ALWAYS:
            why.append(f"policy: {d.action} always needs human review")
        if hard_stop:
            why.append("deterministic hard stop")
        if (critique or {}).get("contradictions"):
            why.append("critique flagged contradictions")
        if d.confidence_band != "high":
            why.append(f"confidence {d.confidence_band}")
        if why:
            results.append(GuardrailResult(rule_id="FORCE_HITL", passed=True, forced_hitl=True,
                                           modified_field="hitl_status", detail="; ".join(why)))
            d.hitl_status, d.status = "escalated", "pending_hitl"
        else:
            ok("HITL_ROUTE", "high confidence, low-risk action -> auto-approved")
            d.hitl_status, d.status = "auto_approved", "final"

    d.guardrail_results, d.updated_at = results, datetime.now(timezone.utc)
    return d