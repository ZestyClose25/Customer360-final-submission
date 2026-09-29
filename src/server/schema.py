from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime

class CustomerProfile(BaseModel):
    customer_id: str

    profile: dict[str, Any] = Field(default_factory=dict)
    accounts: list[dict[str, Any]] = Field(default_factory=list)

# Events (single source of truth lives in memory.schema)
from memory.schema import Event

class EventIngestion(BaseModel):
    event: Event
    customer: CustomerProfile
    is_live: bool


# Agents
class AgentStatus(BaseModel):
    agent_name : str
    status : str

    customer_id : str | None=None
    current_task : str | None=None

    timestamp : datetime

class AgentLog(BaseModel):
    agent_name : str

    customer_id : str | None=None
    event_id : str | None=None

    log_type : str
    message : str

    timestamp : datetime

class AgentDecision(BaseModel):
    decision_id : str
    customer_id : str

    agent_name : str

    inferred_state : str
    confidence_band : str

    action : str
    action_subtype : str | None = None

    hitl_status : str

    notes : str | None=None

    timestamp : datetime

# HITL
class HITLRequest(BaseModel):
    request_id : str
    customer_id : str
    decision_id : str | None=None

    reason : str
    proposed_action : str
    context : dict[str, Any] = Field(default_factory=dict)

    created_at : datetime

class HITLResponse(BaseModel):
    request_id : str

    status : str
    modified_action : str | None=None
    human_notes : str | None=None

    responded_at : str


class HITLDecisionCommand(BaseModel):
    request_id: str
    reviewer: str = "human"
    human_notes: str | None = None
    modified_action: str | None = None
    modified_action_subtype: str | None = None


class HITLAskCommand(BaseModel):
    request_id: str
    question: str = "why is this pending approval?"