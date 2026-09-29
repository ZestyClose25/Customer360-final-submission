import time
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from agents.specialists.config import ALLOWED_HANDOFFS, SOURCE_TO_AGENTS
from graph.state import State
from observe import emit

AgentName = Literal["transaction", "support", "investment", "compliance", "engagement"]


class ManagerDecisionModel(BaseModel):
    next_agents: list[AgentName] = Field(
        description="Specialist agents that should analyze the current event/customer state"
    )
    reason: str = Field(description="Brief explanation of why these specialists are relevant")


MANAGER_SYSTEM_PROMPT = """
You are the Manager Agent in an Agentic Customer 360 system.

Your job is NOT to analyze the customer deeply.

Your job is to decide which specialist agents should be activated
based on the newly received event and the current customer state.

Available specialists:

- transaction:
  Financial transactions, transfers, payments, cashflow,
  unusual financial activity.

- engagement:
  Customer usage, web/app activity, engagement changes,
  behavioral patterns.

- support:
  Support interactions, complaints, sentiment,
  customer-reported problems.

- investment:
  Brokerage, investment, wealth-management activity,
  investment-related opportunities or anomalies.

- compliance:
  KYC, fraud, takeover, suspicious activity,
  compliance-related concerns.

Rules:

1. Activate only relevant specialists.
2. Do NOT activate every specialist by default.
3. Multiple specialists can be selected.
4. If the event provides no meaningful reason for a specialist,
   do not activate it.
5. Do not infer the final life event.
6. Do not decide the final customer action.
7. Keep the reason concise.
8. You are given a deterministic PRIOR (specialists normally woken by this
   event's source system). Follow it unless the event content clearly says
   otherwise; you may return an empty list for a clearly routine event.
"""

MAX_PAYLOAD_CHARS = 1500


def _compact_state(memory) -> str:
    """Small summary of the customer state -- NOT the full memory dump."""
    by_source = Counter(e.source_system for e in memory.recent_events[-30:])
    findings = [
        f"{f.agent_name}/{f.finding_type} ({f.confidence_band}): {f.finding[:160]}"
        for f in memory.agent_findings[-5:]
    ]
    return (
        f"recent events by source (last 30): {dict(by_source)}\n"
        f"total events in working memory: {len(memory.recent_events)}\n"
        f"latest findings:\n- " + "\n- ".join(findings or ["(none yet)"])
    )


async def run_manager(state: State, llm) -> dict:
    """Returns a PARTIAL state update (never the whole state)."""
    rnd = state.get("manager_round", 0)
    completed = set(state.get("completed_agents") or [])

    # ---- round 0: deterministic prior + LLM routing ----
    if rnd == 0:
        event = state["event"]
        memory = state["working_memory"]
        prior = SOURCE_TO_AGENTS.get(event.source_system, [])
        if not prior:  # deterministic gate: unknown source -> nobody wakes up
            return {
                "manager_decisions": {"next_agents": [], "reason": "no specialist maps to this source"},
                "manager_round": rnd + 1,
            }

        payload = str(event.payload)[:MAX_PAYLOAD_CHARS]
        user_prompt = f"""
NEW EVENT:
event_id={event.event_id} source_system={event.source_system} event_type={event.event_type}
event_time={event.event_time.isoformat()} payload={payload}

DETERMINISTIC PRIOR (specialists normally woken by this source): {prior}

CURRENT CUSTOMER STATE (summary):
{_compact_state(memory)}

Determine which specialist agents should be activated.
"""
        ids = dict(customer_id=event.customer_id, event_id=event.event_id)
        emit("llm_prompt", agent="manager", level="debug",
             message=f"routing prompt ({len(user_prompt)} chars)",
             data={"system": MANAGER_SYSTEM_PROMPT, "user": user_prompt, "prior": prior}, **ids)
        t0 = time.perf_counter()
        decision = await llm.agenerate_structured(
            system_prompt=MANAGER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ManagerDecisionModel,
        )
        emit("llm_response", agent="manager", level="debug",
             message=f"routing response [{time.perf_counter() - t0:.1f}s]",
             data={"response": decision.model_dump(mode="json"),
                   "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)
        agents = list(dict.fromkeys(decision.next_agents))
        return {
            "manager_decisions": {"next_agents": agents, "reason": decision.reason},
            "manager_round": rnd + 1,
        }

    # ---- rounds >= 1: deterministic, follow validated handoffs (no LLM call) ----
    agents: list[str] = []
    ev = state["event"]
    for h in state.get("handoff_requests") or []:
        ok = h.to_agent in ALLOWED_HANDOFFS.get(h.from_agent, [])
        if ok and h.to_agent not in completed and h.to_agent not in agents:
            agents.append(h.to_agent)
            why = "accepted"
        else:
            why = ("not allowed by ALLOWED_HANDOFFS" if not ok
                   else "target already ran this event" if h.to_agent in completed
                   else "duplicate request")
        emit("handoff_review", agent="manager",
             message=f"{h.from_agent} -> {h.to_agent}: {why}",
             customer_id=ev.customer_id, event_id=ev.event_id,
             data={"from": h.from_agent, "to": h.to_agent, "verdict": why, "reason": h.reason})
    # critique may ask for a specialist that hasn't run (critique is not a specialist, so it
    # isn't in ALLOWED_HANDOFFS; it may request any specialist that hasn't run yet)
    for r in state.get("critique_requests") or []:
        to = r["to_agent"]
        if to in completed or to in agents:
            why = "target already ran this event" if to in completed else "duplicate request"
        else:
            agents.append(to)
            why = "accepted"
        emit("handoff_review", agent="manager",
             message=f"critique -> {to}: {why}",
             customer_id=ev.customer_id, event_id=ev.event_id,
             data={"from": "critique", "to": to, "verdict": why, "reason": r["reason"]})
    return {
        "manager_decisions": {"next_agents": agents, "reason": "follow specialist handoffs / critique requests"},
        "manager_round": rnd + 1,
    }
