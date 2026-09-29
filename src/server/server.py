# Run from the `src` folder:   uvicorn server.server:app --reload --port 8000
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from graph.build import customer360_graph
from hitl import (
    answer_hitl_question,
    decide_hitl_request,
    ensure_pending_hitl_request,
    get_hitl_request,
    list_hitl_requests,
    trace_for_hitl_request,
)
from observe import RUN_ID, emit, read_since, trace_for_event
from server.schema import EventIngestion, HITLAskCommand, HITLDecisionCommand


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Customer360 server",
    description="Shared source of truth for the Customer360 agents system",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/trace")
def get_trace(after: int = 0, limit: int = 500):
    """Polled by the agent-monitor CLI: trace events with id > `after`."""
    return read_since(after=after, limit=max(1, min(limit, 2000)))


@app.get("/trace/event/{event_id}")
def get_event_trace(event_id: str, run_id: str | None = None):
    """Everything stored about how one event was processed (prompts, responses, findings...)."""
    return {"run_id": run_id or RUN_ID, "event_id": event_id,
            "trace": trace_for_event(event_id, run_id)}


@app.get("/hitl/pending")
def get_pending_hitl(limit: int = 50):
    return {"pending": list_hitl_requests(status="pending", limit=limit)}


@app.get("/hitl/request/{request_id}")
def get_hitl_request_by_id(request_id: str):
    req = get_hitl_request(request_id)
    if not req:
        raise HTTPException(status_code=404, detail=f"unknown request_id: {request_id}")
    return req


@app.get("/hitl/trace/{request_id}")
def get_hitl_trace(request_id: str, limit: int = 100):
    try:
        trace = trace_for_hitl_request(request_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"request_id": request_id, "trace": trace[: max(1, min(limit, 2000))]}


@app.post("/hitl/approve")
def approve_hitl(cmd: HITLDecisionCommand):
    try:
        return decide_hitl_request(
            request_id=cmd.request_id,
            verdict="approve",
            reviewer=cmd.reviewer,
            human_notes=cmd.human_notes,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.post("/hitl/reject")
def reject_hitl(cmd: HITLDecisionCommand):
    try:
        return decide_hitl_request(
            request_id=cmd.request_id,
            verdict="reject",
            reviewer=cmd.reviewer,
            human_notes=cmd.human_notes,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.post("/hitl/modify")
def modify_hitl(cmd: HITLDecisionCommand):
    try:
        return decide_hitl_request(
            request_id=cmd.request_id,
            verdict="modify",
            reviewer=cmd.reviewer,
            human_notes=cmd.human_notes,
            modified_action=cmd.modified_action,
            modified_action_subtype=cmd.modified_action_subtype,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.post("/hitl/ask")
def ask_hitl(cmd: HITLAskCommand):
    try:
        return answer_hitl_question(cmd.request_id, cmd.question)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/events")
async def receive_event(ingestion: EventIngestion):
    event = ingestion.event
    tag = "EVENT" if ingestion.is_live else "HISTORY"
    print(f"[{tag}] {event.event_id} | {event.customer_id} | {event.source_system}/{event.event_type}")

    emit(
        "event_received" if ingestion.is_live else "history_ingested",
        agent="system",
        message=f"{event.source_system}/{event.event_type}",
        customer_id=event.customer_id,
        event_id=event.event_id,
        level="info" if ingestion.is_live else "debug",
        data={
            "source_system": event.source_system,
            "event_type": event.event_type,
            "event_time": event.event_time.isoformat(),
            "account_id": event.account_id,
            "payload": event.payload,
        },
    )

    config = {"configurable": {"thread_id": event.customer_id}}
    t0 = time.perf_counter()
    try:
        result = await customer360_graph.ainvoke(
            {
                "event": event,
                "event_source": "live" if ingestion.is_live else "history",
            },
            config=config,
        )
    except Exception as e:
        traceback.print_exc()
        emit("error", agent="system", message=f"graph failed: {e!r}",
             customer_id=event.customer_id, event_id=event.event_id)
        return {"status": "error", "event_id": event.event_id, "error": repr(e)}

    memory = result["working_memory"]
    if not ingestion.is_live:
        return {"status": "received", "event_id": event.event_id, "is_live": False}

    hitl_request = None
    decision_payload = memory.current_decision.model_dump(mode="json") if memory.current_decision else None
    if decision_payload and decision_payload.get("status") == "pending_hitl":
        hitl_request = ensure_pending_hitl_request(
            run_id=RUN_ID,
            customer_id=memory.customer_id,
            event_id=event.event_id,
            decision=decision_payload,
        )
        if hitl_request:
            emit(
                "hitl_request",
                agent="hitl",
                message=f"pending review {hitl_request['request_id']}",
                customer_id=memory.customer_id,
                event_id=event.event_id,
                data={
                    "request_id": hitl_request["request_id"],
                    "action": decision_payload.get("action"),
                    "reason": hitl_request.get("reason"),
                },
            )

    emit(
        "event_done",
        agent="system",
        message=f"processed in {time.perf_counter() - t0:.1f}s",
        customer_id=event.customer_id,
        event_id=event.event_id,
        data={
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "state_version": memory.state_version,
            "routed_to": result.get("completed_agents") or [],
            "manager_rounds": result.get("manager_round"),
            "n_findings": len(result.get("agent_findings") or []),
            "life_event": ((result.get("life_inference") or {}).get("hypotheses") or [{}])[0].get("life_event"),
            "gate": (result.get("gate_decision") or {}).get("route"),
        },
    )

    return {
        "status": "processed",
        "event_id": event.event_id,
        "customer_id": memory.customer_id,
        "state_version": memory.state_version,
        "routed_to": result.get("completed_agents") or [],
        "manager_rounds": result.get("manager_round"),
        "new_findings": [f.model_dump(mode="json") for f in (result.get("agent_findings") or [])],
        "critique": result.get("critique"),
        "life_inference": result.get("life_inference"),
        "decision": decision_payload,
        "hitl_request": hitl_request,
    }
