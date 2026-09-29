from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime

Confidence = Literal["low", "medium", "high"]

InferredState = Literal[
    "no_significant_event", "new_child_life_event", "marriage_or_relationship_change",
    "job_change_or_promotion", "job_loss_or_income_disruption", "medical_hardship",
    "financial_distress_general", "relocation", "retirement_transition",
    "wealth_growth_or_windfall", "potential_fraud_or_takeover",
    "elder_vulnerability_or_scam_risk", "churn_risk", "small_business_cashflow_event",
]
 
Action = Literal[
    "no_action", "proactive_retention_outreach", "relationship_manager_escalation",
    "personalized_offer", "support_intervention", "compliance_fraud_hold",
]
 
HitlStatus = Literal["auto_approved", "escalated", "human_approved", "human_rejected", "human_modified"]
Status = Literal["active", "resolved", "superseded", "dismissed"]
 
AgentName = Literal[
    "transaction", "engagement", "support", "investment", "compliance",
    "life_event", "synthesis", "intervention", "critique",
]
SpecialistName = Literal["transaction", "engagement", "support", "investment", "compliance"]

AGENT_SOURCE_ACCESS: Dict[str, List[str]] = {
    "transaction": ["card_payments", "instant_payments", "ach_wire", "core_banking_ledger"],
    "engagement":  ["web_app_events"],
    "support":     ["support_logs", "social_signal_consented"],
    "investment":  ["trading_brokerage"],
    "compliance":  ["card_payments", "instant_payments", "ach_wire", "core_banking_ledger", "loan_kyc"],
}

class Event(BaseModel):
    event_id : str
    event_time : datetime
    ingestion_time : datetime

    customer_id : str
    account_id : str | None=None

    source_system : str
    event_type : str

    schema_version : str="1.0"

    payload : dict[str, Any] = Field(default_factory=dict)

# -------------- Working Memory -----------------
class RecentEvent(BaseModel):
    event_id : str
    event_time: datetime
    ingestion_time : datetime
    event_type : str
    source_system : str
    account_id : Optional[str]
    payload : Dict[str, Any]

class AgentFinding(BaseModel):
    agent_name : str
    finding_type : str
    finding : str
    confidence_band : Confidence
    evidence_event_ids : List[str] = Field(default_factory=list)
    created_at : datetime
    updated_at : datetime
    status : str = "active"

class ActiveSituation(BaseModel):
    inferred_state : InferredState
    confidence_band : Confidence
    evidence_event_ids : List[str] = Field(default_factory=list)
    first_detected_at : datetime
    last_updated_at : datetime
    status : str = "active"

class ActiveRisk(BaseModel):
    risk_type : str
    description : str
    confidence_band : Confidence
    evidence_event_ids : List[str] = Field(default_factory=list)
    status : str = "active"

class ActiveOppurtunity(BaseModel):
    opportunity_type : str
    description : str
    confidence_band : Confidence
    evidence_event_ids : List[str] = Field(default_factory=list)
    status : str="active"

class CurrentCase(BaseModel):
    case_id : str
    case_type : str
    trigger_event_ids : List[str] = Field(default_factory=list)
    participating_agents : List[str] = Field(default_factory=list)
    status : str = "active"
    started_at : datetime
    last_updated_at : datetime

class Explanation(BaseModel):
    summary: str
    signals: List[str] = Field(default_factory=list)
    evidence_event_ids: List[str] = Field(default_factory=list)
    memory_refs: List[str] = Field(default_factory=list)      # episode ids
    policy_refs: List[str] = Field(default_factory=list)      # knowledge ids
    finding_ids: List[str] = Field(default_factory=list)

class GuardrailResult(BaseModel):
    rule_id: str
    passed: bool
    detail: str
    forced_hitl: bool = False
    modified_field: Optional[str] = None

class CurrentDecision(BaseModel):
    inferred_state : InferredState = "no_significant_event"
    confidence_band : Confidence = "low"
    action : Action = "no_action"
    action_subtype : Optional[str] = None
    evidence_event_ids: List[str] = Field(default_factory=list)
    hitl_status: HitlStatus = "auto_approved"
    guardrail_results: List[GuardrailResult] = Field(default_factory=list)
    notes: str = ""
    explanation: Optional[Explanation] = None
    status: Literal["draft", "pending_hitl", "final"] = "draft"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class WorkingMemory(BaseModel):
    customer_id : str
    account_ids : List[str] = Field(default_factory=list)  
    recent_events : List[RecentEvent] = Field(default_factory=list)
    agent_findings : List[AgentFinding] = Field(default_factory=list)
    active_situation : Optional[ActiveSituation] = None
    active_risks : List[ActiveRisk] = Field(default_factory=list)
    active_opportunities : List[ActiveOppurtunity] = Field(default_factory=list)
    current_case : Optional[CurrentCase] = None
    current_decision : Optional[CurrentDecision] = None
    latest_event_time : Optional[datetime] = None
    latest_ingestion_time : datetime
    last_updated_at : datetime
    events_since_analysis : int = 0

    state_version : int = 1

# ------------------ Episodic Memory ---------------------
class EpisodeOutcome(BaseModel):
    status: str
    description: Optional[str] = None
    observed_at: Optional[datetime] = None


class EpisodeAction(BaseModel):
    proposed_action: Optional[Action] = None
    final_action: Optional[Action] = None
    action_subtype: Optional[str] = None
    hitl_required: bool = False
    hitl_status: Optional[HitlStatus] = None
    human_modified_action: Optional[str] = None


class Episode(BaseModel):
    episode_id: str
    customer_id: str
    episode_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    situation: str
    inferred_state: Optional[InferredState] = None
    confidence_band: Optional[Confidence] = None
    evidence_event_ids: List[str] = Field(default_factory=list)
    participating_agents: List[str] = Field(default_factory=list)
    action: Optional[EpisodeAction] = None
    outcome: Optional[EpisodeOutcome] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    last_updated_at: datetime


class EpisodicMemory(BaseModel):
    customer_id: str
    episodes: List[Episode] = Field(default_factory=list)
    last_updated_at: datetime

# --------------- Semantic / Shared Memory --------------------
class KnowledgeSource(BaseModel):
    source_id: str
    source_type: str
    source_reference: Optional[str] = None


class KnowledgeItem(BaseModel):
    knowledge_id: str
    knowledge_type: str
    title: str
    content: str
    conditions: Dict[str, Any] = Field(default_factory=dict)
    applicable_actions: List[str] = Field(default_factory=list)
    source: Optional[KnowledgeSource] = None
    tags: List[str] = Field(default_factory=list)
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    last_updated_at: datetime


class SemanticMemory(BaseModel):
    knowledge_items: List[KnowledgeItem] = Field(default_factory=list)
    last_updated_at: datetime


class AgentContext(BaseModel):
    working_memory: WorkingMemory
    episodic_memory: EpisodicMemory
    semantic_memory: SemanticMemory

# ------------------------------------------------------
class HandoffRequest(BaseModel):
    from_agent: str
    to_agent: str
    reason: str
    evidence_event_ids: List[str] = Field(default_factory=list)

def append_list(left, right):
    return (left or []) + (right or [])

def union_list(left, right):
    out = list(left or [])
    for x in right or []:
        if x not in out:
            out.append(x)

    return out


class AgentGraphState(TypedDict, total=False):
    run_id: str
    customer_id: str
    trigger_event_ids: List[str]

    # loaded once at the start, read-only after that
    working_memory: WorkingMemory
    episodic_memory: EpisodicMemory
    semantic_memory: SemanticMemory

    # Manager's routing decision (single writer = Manager, so no reducer needed)
    pending_agents: List[str]
    next_step: str

    # written by MANY specialists in parallel -> needs a reducer
    agent_findings: Annotated[List[AgentFinding], append_list]
    completed_agents: Annotated[List[str], union_list]
    handoff_requests: Annotated[List[HandoffRequest], append_list]

    # single writer each, later in the pipeline
    active_situation: Optional[ActiveSituation]
    current_decision: Optional[CurrentDecision]