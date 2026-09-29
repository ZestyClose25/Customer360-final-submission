import json
import time
from datetime import datetime, timezone
from typing import List, Literal

from pydantic import BaseModel, Field

from agents.specialists.config import AGENT_SOURCE_ACCESS, ALLOWED_HANDOFFS
from llm.client import get_llm
from observe import emit
from memory.schema import AgentFinding, Confidence, HandoffRequest, RecentEvent

from tools import ToolInput, run_tools, replay_now

MAX_CONTEXT_EVENTS = 25   
MAX_PRIOR_FINDINGS = 6    


class FindingDraft(BaseModel):
    finding_type: str = Field(description="short snake_case label, e.g. income_drop, unusual_transfer")
    finding: str = Field(description="1-3 sentence factual finding grounded in the events")
    confidence_band: Confidence
    evidence_event_ids: List[str] = Field(default_factory=list, description="event_ids supporting the finding")


class HandoffDraft(BaseModel):
    to_agent: Literal["transaction", "engagement", "support", "investment", "compliance"]
    reason: str
    evidence_event_ids: List[str] = Field(default_factory=list)


class SpecialistOutput(BaseModel):
    summary: str = Field(description="one line summary of what you found")
    findings: List[FindingDraft] = Field(default_factory=list)
    handoffs: List[HandoffDraft] = Field(default_factory=list)


class BaseAgent:
    key: str = ""            # graph node name: transaction / engagement / ...
    name: str = ""           # display name
    SYSTEM_PROMPT: str = ""

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def llm(self):
        return self._llm or get_llm()

    @property
    def allowed_sources(self) -> list[str]:
        return AGENT_SOURCE_ACCESS.get(self.key, [])

    async def analyze(
        self,
        customer_id: str,
        events: List[RecentEvent],          # already source-filtered + account-masked
        trigger_event_id: str,
        prior_findings: List[AgentFinding],
        tool_results: dict | None = None
    ) -> SpecialistOutput:
        if not events:
            return SpecialistOutput(summary="no events in my sources", findings=[])

        ev = [e.model_dump(mode="json") for e in events[-MAX_CONTEXT_EVENTS:]]
        prior = [
            {
                "agent": f.agent_name, "type": f.finding_type, "finding": f.finding,
                "confidence": f.confidence_band, "evidence": f.evidence_event_ids,
            }
            for f in prior_findings[-MAX_PRIOR_FINDINGS:]
        ]
        allowed = ALLOWED_HANDOFFS.get(self.key, [])

        user_prompt = f"""
Customer: {customer_id}
Newest event that triggered this analysis: {trigger_event_id}

Events from your data sources (oldest -> newest):
{json.dumps(ev, indent=1)}

Earlier findings from all agents (context only):
{json.dumps(prior, indent=1)}

Tool results (deterministic, computed from the events above; prefer these numbers over your own arithmetic):
{json.dumps(tool_results or {{}}, indent=1, default=str)}

Task: report meaningful observable findings, focusing on what is NEW or CHANGED
because of the newest event. If nothing meaningful, return an empty findings list.
evidence_event_ids MUST come from the event_ids above.
You may request a handoff only to: {allowed}. Do so only if another specialist's
data is needed to interpret what you see.
"""
        ids = dict(customer_id=customer_id, event_id=trigger_event_id)
        emit("llm_prompt", agent=self.key, level="debug",
             message=f"prompt ({len(user_prompt)} chars, {len(ev)} events, {len(prior)} prior findings)",
             data={"system": self.SYSTEM_PROMPT, "user": user_prompt}, **ids)
        t0 = time.perf_counter()
        out = await self.llm.agenerate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=SpecialistOutput,
        )
        emit("llm_response", agent=self.key, level="debug",
             message=f"response [{time.perf_counter() - t0:.1f}s]",
             data={"response": out.model_dump(mode="json"),
                   "elapsed_s": round(time.perf_counter() - t0, 2)}, **ids)
        return out


def make_specialist_node(agent: BaseAgent):
    key = agent.key

    async def node(state):
        from graph.context import build_context   # lazy: avoids circular import

        wm = state["working_memory"]
        ctx = build_context(key, state)            # scoped + masked view
        t0 = time.perf_counter()
        ids = dict(customer_id=wm.customer_id, event_id=state["event"].event_id)
        emit("agent_start", agent=key,
             message=f"analyzing {len(ctx)} event(s) from my sources",
             data={"n_context_events": len(ctx), "sources": agent.allowed_sources,
                   "n_prior_findings": len(wm.agent_findings)}, **ids)
        trigger = next((e for e in ctx if e.event_id == state["event"].event_id), None)
        try:
            tool_results = run_tools(
                key,
                ToolInput(customer_id=wm.customer_id, trigger=trigger, events=ctx,
                      findings=wm.agent_findings, now=replay_now(wm.recent_events, state["event"].event_time)),
                ids,
            )
            out = await agent.analyze(
                customer_id=wm.customer_id,
                events=ctx,
                trigger_event_id=state["event"].event_id,
                prior_findings=wm.agent_findings,
                tool_results=tool_results,
            )
        except Exception as e:
            print(f"   ! {key} agent ERROR: {e!r}")
            emit("agent_error", agent=key, message=repr(e), **ids)
            return {"completed_agents": [key]}

        now = datetime.now(timezone.utc)
        valid_ids = {e.event_id for e in wm.recent_events}
        findings = [
            AgentFinding(
                agent_name=key,
                finding_type=d.finding_type,
                finding=d.finding,
                confidence_band=d.confidence_band,
                evidence_event_ids=[i for i in d.evidence_event_ids if i in valid_ids],
                created_at=now,
                updated_at=now,
            )
            for d in out.findings
        ]
        allowed_to = ALLOWED_HANDOFFS.get(key, [])
        handoffs = [
            HandoffRequest(from_agent=key, to_agent=h.to_agent, reason=h.reason,
                           evidence_event_ids=h.evidence_event_ids)
            for h in out.handoffs
            if h.to_agent != key and h.to_agent in allowed_to
        ]
        for h in out.handoffs:      # rejected handoffs are traced, not silently dropped
            if h.to_agent == key or h.to_agent not in allowed_to:
                emit("handoff_rejected", agent=key,
                     message=f"{key} -> {h.to_agent} rejected (allowed: {allowed_to})",
                     data={"from": key, "to": h.to_agent, "reason": h.reason}, **ids)
        dropped_ids = [i for d in out.findings for i in d.evidence_event_ids if i not in valid_ids]
        print(f"   → {key}: {len(findings)} finding(s), {len(handoffs)} handoff(s) "
              f"[{time.perf_counter() - t0:.1f}s] {out.summary}")
        for f in findings:
            emit("agent_finding", agent=key, message=f.finding,
                 data={"finding_type": f.finding_type, "confidence_band": f.confidence_band,
                       "evidence_event_ids": f.evidence_event_ids}, **ids)
        for h in handoffs:
            emit("handoff_request", agent=key, message=h.reason,
                 data={"from": key, "to": h.to_agent, "reason": h.reason,
                       "evidence_event_ids": h.evidence_event_ids}, **ids)
        emit("agent_done", agent=key, message=out.summary,
             data={"n_findings": len(findings), "n_handoffs": len(handoffs),
                   "elapsed_s": round(time.perf_counter() - t0, 2),
                   "invalid_evidence_ids_dropped": dropped_ids}, **ids)

        upd: dict = {"completed_agents": [key]}
        if findings:
            upd["agent_findings"] = findings
        if handoffs:
            upd["handoff_requests"] = handoffs
        return upd

    node.__name__ = f"{key}_node"
    return node
