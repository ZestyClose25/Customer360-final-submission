# Customer360: Agentic Proactive Intervention Desk

An autonomous, multi-agent system designed to act as an ambient Customer 360 desk. This system continuously monitors streaming customer data (transactions, support tickets, engagement) to build evolving customer context, infer life events, and proactively propose interventions. 

Built to enterprise safety standards, the architecture features strict human-in-the-loop (HITL) checkpoints, explicit multi-agent coordination, deterministic guardrails, and real-time observability.

## Implemented Architecture

This repository contains a functional multi-agent system utilizing LangGraph for orchestration. The implemented components include:

*   **Multi-Agent Swarm (Specialists):** Domain-specific agents (`transaction`, `engagement`, `support`, `investment`, `compliance`) that analyze raw events in parallel using targeted toolsets.
*   **Graph Orchestrator:** A sophisticated state machine coordinating the workflow through specific nodes: `gate` (routing), `manager` (agent orchestration), `critique` (quality assurance), `life_inference`, `intervention`, and `guardrails` (deterministic safety stops).
*   **Agent Memory:** Short-term working memory and episodic per-customer event histories injected directly into the LLM context to ground decisions.
*   **Observability & Traceability:** A live event bus (`observe/bus.py`) that broadcasts all agent thoughts, tool executions, and state changes to a dedicated terminal monitor.
*   **Human-In-The-Loop (HITL):** A dedicated CLI allowing human reviewers to approve, reject, or modify high-stakes agent proposals before execution.
*   **Live Stream Simulation:** A testing streamer that ingests historical customer seeds and sequentially fires live events to trigger agent workflows.

*(Note: Out-of-order event management, semantic memory, vector databases for policies, and CRM notes are excluded from this baseline implementation.)*

---

## Setup Guide

As a developer cloning this repository, follow these steps to set up your local environment, install dependencies from configuration files (`pyproject.toml` / `requirements.txt`), and configure your environment variables.

### 1. Clone & Environment Setup
Open your terminal in the root directory of the cloned repository and set up an isolated virtual environment:

```bash
# Create a virtual environment using uv or standard venv
uv venv .venv

# Activate the virtual environment
# On Windows (Command Prompt / PowerShell):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -e .
# Or via pip using requirements:
pip install -r src/requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory (referencing `src/.env`) and add your required environment keys (such as your Groq API key):
```env
GROQ_API_KEY=your_actual_api_key_here
```

---

## Running the Services

To run the full Customer360 simulation, you will need to open **four separate terminal windows**. Ensure the virtual environment is activated in every window. 

The execution **sequence is critical**: start the server first, followed by the monitor, then the streamer, and finally the HITL CLI.

### Terminal 1: Start the Backend Server
The server hosts the LangGraph orchestration pipeline and the trace event bus.
```bash
cd src
uvicorn server.server:app --reload
```

### Terminal 2: Start the Agent Monitor
The monitor provides a live narrative view into the agentic layer, rendering internal reasoning, tool calls, and LLM responses in real-time.
```bash
cd src\monitor
uv run monitor.py
```

### Terminal 3: Start the Event Streamer
The streamer acts as the environment, loading historical customer seed data and streaming live asynchronous events to trigger the agent workflows.
```bash
cd src\streamer
uv run stream.py
```
*(Follow the interactive prompt in the streamer terminal to select a scenario).*

### Terminal 4: Start the HITL (Human-In-The-Loop) CLI
The HITL console serves as the final authorization checkpoint, allowing you to review, approve, reject, or modify agent-proposed interventions.
```bash
cd src\hitl
uv run cli.py
```

Once all four services are active, the system will autonomously ingest streaming events, execute multi-agent reasoning loops, and queue up interventions for your review.