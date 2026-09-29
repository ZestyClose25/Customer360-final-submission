# Tools for the critique and life-inference agents. They read the shared board, not raw sources.
from collections import defaultdict

from tools.base import ToolInput, register

SRC = "shared state board: this event's findings + working-memory events"


@register("critique", reads=SRC)
def verify_evidence(inp: ToolInput) -> dict:
    """For each finding under review: are the cited event ids real, from which sources, how old, single-event?"""
    by_id = {e.event_id: e for e in inp.events}
    rows = []
    for i, f in enumerate(inp.findings):
        ev = [by_id[x] for x in f.evidence_event_ids if x in by_id]
        rows.append({"finding_index": i, "agent": f.agent_name, "n_cited": len(f.evidence_event_ids),
                     "n_valid": len(ev), "sources": sorted({e.source_system for e in ev}),
                     "oldest_evidence_age_days": max(((inp.now - e.event_time).days for e in ev), default=None),
                     "single_event_only": len(ev) == 1})
    return {"findings": rows}


@register("critique", "life_inference", reads=SRC)
def correlated_findings(inp: ToolInput) -> dict:
    """Which agents flagged something, and which cited evidence events are shared between agents."""
    by_agent = defaultdict(list)
    ev_agents = defaultdict(set)
    for f in inp.findings:
        by_agent[f.agent_name].append(f.finding_type)
        for x in f.evidence_event_ids:
            ev_agents[x].add(f.agent_name)
    return {"n_agents_flagging": len(by_agent), "by_agent": dict(by_agent),
            "events_cited_by_multiple_agents": {k: sorted(v) for k, v in ev_agents.items() if len(v) > 1}}


@register("life_inference", reads="life_history kept in graph state per customer (thread)")
def life_phase_history(inp: ToolInput) -> dict:
    """The last few life-phase inferences already made for this customer (so they carry forward)."""
    hist = (inp.extra.get("life_history") or [])[-5:]
    return {"n_previous": len(hist), "previous": hist}