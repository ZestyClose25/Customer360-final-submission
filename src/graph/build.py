from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.critique import MAX_CRITIQUE_ROUNDS
from agents.specialists.config import MAX_MANAGER_ROUNDS, SPECIALISTS
from graph.nodes import (
    critique_node,
    life_inference_node,
    compliance_node,
    engagement_node,
    finalize_node,
    gate_node,
    investment_node,
    manager_node,
    support_node,
    transaction_node,
    update_working_memory,
    intervention_node,
    guardrails_node
)
from graph.state import State
from observe import emit

def route_after_gate(state: State):
    return {"agents": "manager", "hard_stop": "guardrails"}.get(state["gate_decision"]["route"], "finalize")


def route_after_memory_update(state: State):
    return "history_done" if state["event_source"] == "history" else "gate"


def _after_specialists(state: State) -> str:
    """Nobody left to wake: critique any unreviewed findings, then infer, else finish."""
    findings = state.get("agent_findings") or []
    reviewed = set((state.get("critique") or {}).get("reviewed_agents") or [])
    if any(f.agent_name not in reviewed for f in findings) and state.get("critique_round", 0) < MAX_CRITIQUE_ROUNDS:
        return "critique"
    return "life_inference" if findings else "finalize"


def route_after_manager(state: State):
    decision = state.get("manager_decisions")
    completed = state.get("completed_agents") or []
    agents = [a for a in (decision["next_agents"] if decision else []) if a not in completed]

    capped = state.get("manager_round", 0) > MAX_MANAGER_ROUNDS
    if agents and capped:
        emit("loop_guard", agent="manager", level="info",
             message=f"round cap ({MAX_MANAGER_ROUNDS}) hit; not dispatching {agents}",
             customer_id=state["event"].customer_id, event_id=state["event"].event_id,
             data={"undispatched": agents})
    if not agents or capped:
        return _after_specialists(state)

    return [Send(a, state) for a in agents]      # parallel fan-out


def route_after_critique(state: State):
    return "manager" if state.get("critique_requests") else "life_inference"


_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("memory.schema", name)
        for name in (
            "Event", "RecentEvent", "WorkingMemory", "AgentFinding", "HandoffRequest",
            "ActiveSituation", "ActiveRisk", "ActiveOppurtunity", "CurrentCase",
            "CurrentDecision", "GuardrailResult", "Explanation",
        )
    ]
)


def build_graph():
    builder = StateGraph(State)

    builder.add_node("update_working_memory", update_working_memory)
    builder.add_node("gate", gate_node)
    builder.add_node("manager", manager_node)
    builder.add_node("transaction", transaction_node)
    builder.add_node("engagement", engagement_node)
    builder.add_node("support", support_node)
    builder.add_node("investment", investment_node)
    builder.add_node("compliance", compliance_node)
    builder.add_node("critique", critique_node)
    builder.add_node("life_inference", life_inference_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("intervention", intervention_node)
    builder.add_node("guardrails", guardrails_node)

    builder.add_edge(START, "update_working_memory")

    builder.add_conditional_edges(
        "gate", route_after_gate,
        {"manager": "manager", "guardrails": "guardrails", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "update_working_memory",
        route_after_memory_update,
        {"history_done": END, "gate": "gate"},
    )
    builder.add_conditional_edges(
        "manager",
        route_after_manager,
        [*SPECIALISTS, "critique", "life_inference", "finalize"],
    )
    builder.add_conditional_edges(
        "critique", route_after_critique, {"manager": "manager", "life_inference": "life_inference"})
    builder.add_edge("life_inference", "intervention")
    builder.add_edge("intervention", "guardrails")
    builder.add_edge("guardrails", "finalize")
    for agent in SPECIALISTS:
        builder.add_edge(agent, "manager")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=InMemorySaver(serde=_SERDE))


customer360_graph = build_graph()