from typing import List

from memory.schema import RecentEvent
from graph.state import State
from agents.specialists.config import AGENT_SOURCE_ACCESS

def mask_account(account_id: str|None) -> str | None:
    if not account_id: return None
    return "***" + account_id[-3:]

def build_context(agent_name: str, state: State) -> List[RecentEvent]:
    memory = state["working_memory"]
    allowed_sources = AGENT_SOURCE_ACCESS.get(agent_name, [])

    return [e.model_copy(update={"account_id": mask_account(e.account_id)}) for e in memory.recent_events if e.source_system in allowed_sources]