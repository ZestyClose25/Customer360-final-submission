from datetime import datetime, timezone

from agents.base import make_specialist_node
from agents.critique import run_critique
from agents.life_inference import run_life_inference
from agents.registry import build_agents
from agents.intervention import run_intervention
from graph.manager import run_manager
from graph.state import State
from llm.client import get_llm
from observe import emit
from memory.schema import RecentEvent, WorkingMemory

from graph.gate import evaluate_gate
from graph.guardrails import apply_guardrails, hard_stop_decision

MAX_RECENT_EVENTS = 80     # working-memory window
MAX_FINDINGS = 30

def guardrails_node(state: State) -> dict:
    wm, event = state["working_memory"], state["event"]
    gate = state.get("gate_decision") or {}
    draft = state.get("proposed_decision")
    if draft is None and gate.get("route") == "hard_stop":
        draft = hard_stop_decision(event, gate)        
    if draft is None:
        return {}

    final = apply_guardrails(
        draft, memory=wm, critique=state.get("critique") or {},
        life_inference=state.get("life_inference") or {},
        life_history=state.get("life_history") or [],
        gate=gate,
        ref_time=max(e.event_time for e in wm.recent_events)
    )
    fired = [g.rule_id for g in final.guardrail_results if not g.passed or g.forced_hitl]
    print(f"   guardrails: {draft.action} -> {final.action} | hitl={final.hitl_status} | fired={fired}")
    emit("guardrails", agent="guardrails",
         message=f"{draft.action} -> {final.action}, hitl={final.hitl_status}",
         customer_id=event.customer_id, event_id=event.event_id,
         data={"proposed_action": draft.action, "final_action": final.action,
               "hitl_status": final.hitl_status, "status": final.status,
               "results": [g.model_dump(mode="json") for g in final.guardrail_results]})
    return {"proposed_decision": final}

async def intervention_node(state: State) -> dict:
    return await run_intervention(state, get_llm())


def update_working_memory(state: State) -> dict:
    event = state["event"]
    prev = state.get("working_memory")          # .get(): first event has no memory yet
    now = datetime.now(timezone.utc)

    if prev is None:
        memory = WorkingMemory(
            customer_id=event.customer_id,
            latest_event_time=event.event_time,
            latest_ingestion_time=event.ingestion_time,
            last_updated_at=now,
        )
    else:
        memory = prev.model_copy(deep=True)

    memory.recent_events.append(
        RecentEvent(
            event_id=event.event_id,
            event_time=event.event_time,
            ingestion_time=event.ingestion_time,
            event_type=event.event_type,
            source_system=event.source_system,
            account_id=event.account_id,
            payload=event.payload,
        )
    )
    memory.recent_events = memory.recent_events[-MAX_RECENT_EVENTS:]

    if event.account_id and event.account_id not in memory.account_ids:
        memory.account_ids.append(event.account_id)

    memory.latest_event_time = event.event_time
    memory.latest_ingestion_time = event.ingestion_time
    memory.last_updated_at = now
    memory.state_version += 1

    print(f"   memory[{event.customer_id}] v{memory.state_version} events={len(memory.recent_events)}")
    emit("memory_updated", agent="memory", level="debug",
         message=f"working memory v{memory.state_version}, {len(memory.recent_events)} events",
         customer_id=event.customer_id, event_id=event.event_id,
         data={"version": memory.state_version, "n_events": len(memory.recent_events)})

    return {
        "working_memory": memory,
        # reset per-event scratch (thread state persists across events)
        "manager_decisions": None,
        "manager_round": 0,
        "completed_agents": None,     # None => reducer resets
        "agent_findings": None,
        "handoff_requests": None,
        # critique / life-inference scratch (plain overwrite keys)
        "reviewed_findings": None,
        "critique": None,
        "critique_requests": None,
        "critique_round": 0,
        "life_inference": None,
        "proposed_decision": None
    }


async def manager_node(state: State) -> dict:
    out = await run_manager(state, get_llm())
    d = out["manager_decisions"]
    print(f"   manager[round {out['manager_round']}] -> {d['next_agents']} ({d['reason'][:80]})")
    emit("manager_decision", agent="manager", message=d["reason"],
         customer_id=state["event"].customer_id, event_id=state["event"].event_id,
         data={"round": out["manager_round"], "next_agents": d["next_agents"], "reason": d["reason"]})
    return out


async def critique_node(state: State) -> dict:
    return await run_critique(state, get_llm())


async def life_inference_node(state: State) -> dict:
    return await run_life_inference(state, get_llm())


def finalize_node(state: State) -> dict:
    """Terminal node ('done'): merge this event's findings into working memory.
    Synthesis / Critique / Intervention / guardrails get inserted before this later."""
    memory = state["working_memory"].model_copy(deep=True)
    new = state.get("agent_findings") or []
    memory.agent_findings = (memory.agent_findings + new)[-MAX_FINDINGS:]
    memory.last_updated_at = datetime.now(timezone.utc)
    memory.state_version += 1
    gate = state.get("gate_decision") or {}
    if state.get("proposed_decision"):
        memory.current_decision = state["proposed_decision"]
    emit("finalize", agent="system",
         message=f"memory v{memory.state_version}, +{len(new)} finding(s), {len(memory.agent_findings)} kept",
         customer_id=memory.customer_id, event_id=state["event"].event_id,
         data={"version": memory.state_version, "new_findings": len(new),
               "total_findings": len(memory.agent_findings), "gate_route": gate.get("route"),
               "life_event": ((state.get("life_inference") or {}).get("hypotheses") or [{}])[0].get("life_event")})
    return {"working_memory": memory}

def gate_node(state: State) -> dict:
    event = state["event"]
    memory = state["working_memory"].model_copy(deep=True)
    d = evaluate_gate(event, memory)
    memory.events_since_analysis = 0 if d["route"] == "agents" else memory.events_since_analysis + 1
    print(f"   gate -> {d['route']} {d['rule_ids']} ({d['reasons'][0][:70]})")
    emit("gate", agent="gate", message=d["reasons"][0],
         customer_id=event.customer_id, event_id=event.event_id,
         data={"route": d["route"], "rule_ids": d["rule_ids"], "reasons": d["reasons"],
               "events_since_analysis": memory.events_since_analysis})
    return {"gate_decision": d, "working_memory": memory}


AGENTS = build_agents()
transaction_node = make_specialist_node(AGENTS["transaction"])
engagement_node = make_specialist_node(AGENTS["engagement"])
support_node = make_specialist_node(AGENTS["support"])
investment_node = make_specialist_node(AGENTS["investment"])
compliance_node = make_specialist_node(AGENTS["compliance"])
