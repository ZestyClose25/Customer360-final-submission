from typing import TypedDict, Literal, Annotated

from memory.schema import WorkingMemory, Event, AgentFinding, HandoffRequest, CurrentDecision

AgentName = Literal[
    "transaction",
    "support",
    "compliance",
    "engagement",
    "investment",
]


class ManagerDecision(TypedDict):
    next_agents: list[AgentName]
    reason: str


# Reducers: passing `None` RESETS the channel. The thread (customer_id) is
# checkpointed across events, so per-event fields must be cleared at the start
# of every event (done in update_working_memory).
def union_list(left: list | None, right: list | None) -> list:
    if right is None:
        return []
    out = list(left or [])
    for x in right:
        if x not in out:
            out.append(x)
    return out


def append_list(left: list | None, right: list | None) -> list:
    if right is None:
        return []
    return (left or []) + right

class GateDecision(TypedDict):
    route: Literal["skip", "agents", "hard_stop"]
    reasons: list[str]
    rule_ids: list[str]

class State(TypedDict, total=False):
    # input (overwritten on every /events call)
    event: Event
    event_source: Literal["history", "live"]

    # persists across events for the same customer (checkpointed)
    working_memory: WorkingMemory | None

    # per-event scratch (reset by update_working_memory)
    manager_decisions: ManagerDecision | None
    manager_round: int
    completed_agents: Annotated[list[str], union_list]
    agent_findings: Annotated[list[AgentFinding], append_list]
    handoff_requests: Annotated[list[HandoffRequest], append_list]
    gate_decision: GateDecision | None
    reviewed_findings: list | None
    critique : dict | None
    critique_requests: list | None
    critique_round: int
    life_inference: dict | None
    life_history: list | None
    proposed_decision: CurrentDecision | None