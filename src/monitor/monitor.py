#!/usr/bin/env python3
"""
Customer360 Agent Monitor -- a live window into the agentic layer.

The server (uvicorn) runs the LangGraph pipeline; every node/agent writes trace events
to the trace bus (src/observe). This CLI just polls  GET {url}/trace  and renders them,
so it can be started/stopped at any time without touching the pipeline.

    python monitor/monitor.py                     # narrative view, tail from now
    python monitor/monitor.py -v                  # + LLM prompts and raw structured responses
    python monitor/monitor.py -vv                 # + full (unclipped) prompts, system prompts, memory updates
    python monitor/monitor.py --from-start        # replay everything still in the server's buffer
    python monitor/monitor.py --customer CUST_00088 --agent transaction compliance
    python monitor/monitor.py --history           # also show history-seed ingestion lines
    python monitor/monitor.py --replay trace.jsonl --speed 4   # replay a saved run (no server needed)

Only needs: requests, rich.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

SPECIALISTS = ["transaction", "engagement", "support", "investment", "compliance"]

COLORS = {
    "transaction": "cyan",
    "engagement": "magenta",
    "support": "green",
    "investment": "blue",
    "compliance": "red",
    "manager": "yellow",
    "critique": "orange3",
    "life_inference": "bright_magenta",
    "gate": "bright_yellow",
    "memory": "grey50",
    "system": "white",
}

# debug-level trace kinds are hidden unless -v (or -vv for the noisier ones)
DEBUG_MIN_VERBOSITY = {"llm_prompt": 1, "llm_response": 1, "memory_updated": 2}

CONF_STYLE = {"high": "bold green", "medium": "yellow", "low": "dim"}
POLL_LIMIT = 500


def color_of(agent: str) -> str:
    return COLORS.get(agent, "white")


def short_payload(payload: dict, limit: int = 150) -> str:
    parts = []
    for k, v in (payload or {}).items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
        parts.append(f"{k}={v}")
    s = " ".join(parts)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def clip(s: str, n: int) -> str:
    return s if n <= 0 or len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"


def local_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S.%f")[:-3]
    except Exception:
        return ts[-12:]


class Monitor:
    def __init__(self, console: Console, verbosity: int = 0, show_history: bool = False,
                 agents: set[str] | None = None, customer: str | None = None):
        self.console = console
        self.verbosity = verbosity
        self.show_history = show_history
        self.agents = agents or set()
        self.customer = customer

        self.status = {a: {"state": "idle", "since": 0.0, "last": None} for a in SPECIALISTS}
        self.counts: Counter = Counter()
        self.current_event: str | None = None
        self.connection = "connecting"

    # ------------------------------------------------------------------ state
    def _update_state(self, rec: dict) -> None:
        kind, agent = rec["kind"], rec["agent"]
        c = self.counts
        if kind == "event_received":
            c["live"] += 1
            for a, st in self.status.items():
                if st["state"] != "running":
                    st["state"] = "idle"
        elif kind == "history_ingested":
            c["history"] += 1
        elif kind == "agent_finding":
            c["findings"] += 1
        elif kind == "handoff_request":
            c["handoffs"] += 1
        elif kind == "llm_response":
            c["llm calls"] += 1
        elif kind in ("error", "agent_error"):
            c["errors"] += 1

        if agent in self.status:
            st = self.status[agent]
            if kind == "agent_start":
                st.update(state="running", since=time.monotonic())
            elif kind == "agent_done":
                st.update(state="done", last=(rec.get("data") or {}).get("elapsed_s"))
            elif kind == "agent_error":
                st.update(state="error")

    def _visible(self, rec: dict) -> bool:
        kind = rec["kind"]
        if kind == "history_ingested":
            return self.show_history
        if rec.get("level") == "debug" and self.verbosity < DEBUG_MIN_VERBOSITY.get(kind, 1):
            return False
        if self.customer and rec.get("customer_id") not in (None, self.customer):
            return False
        if self.agents and rec["agent"] not in self.agents and kind not in ("event_received", "event_done", "error"):
            return False
        return True

    # ---------------------------------------------------------------- output
    def _row(self, rec: dict, tag: str, body, tag_style: str | None = None) -> None:
        g = Table.grid(expand=True, padding=(0, 1))
        g.add_column(no_wrap=True, width=12)
        g.add_column(no_wrap=True, width=13)
        g.add_column(ratio=1)
        g.add_row(Text(local_time(rec["ts"]), style="grey50"),
                  Text(tag.upper(), style=tag_style or f"bold {color_of(rec['agent'])}"),
                  body)
        self.console.print(g)

    def _context_header(self, rec: dict) -> None:
        """When lines of a different event show up (concurrent processing), say whose they are."""
        eid = rec.get("event_id")
        if eid and eid != self.current_event and rec["kind"] != "event_received":
            self.console.print(Rule(f"{eid} | {rec.get('customer_id') or ''}", style="grey37", align="left"))
            self.current_event = eid

    def notice(self, text: str, style: str = "yellow") -> None:
        self.console.print(Text(f"-- {text}", style=style))

    def handle(self, rec: dict) -> None:
        self._update_state(rec)
        if not self._visible(rec):
            return
        self._context_header(rec)
        fn = getattr(self, f"_on_{rec['kind']}", None)
        if fn:
            fn(rec)
        else:  # unknown kinds are still shown -- never hide what the pipeline reports
            self._row(rec, rec["kind"], Text(rec.get("message", "")))

    # -------------------------------------------------------------- renderers
    def _on_event_received(self, rec):
        d = rec.get("data") or {}
        self.current_event = rec["event_id"]
        self.console.print(Rule(
            f" {rec['event_id']} | {rec['customer_id']} | {d.get('source_system')}/{d.get('event_type')} ",
            style="bold white"))
        self._row(rec, "event", Text(short_payload(d.get("payload") or {}), style="white"), "bold white")

    def _on_history_ingested(self, rec):
        d = rec.get("data") or {}
        self._row(rec, "history", Text(f"{rec['event_id']} {d.get('source_system')}/{d.get('event_type')}", style="dim"), "dim")

    def _on_memory_updated(self, rec):
        self._row(rec, "memory", Text(rec["message"], style="dim"))

    def _on_gate(self, rec):
        d = rec["data"]
        route = d["route"]
        style = {"agents": "bold green", "skip": "dim yellow", "hard_stop": "bold red"}.get(route, "white")
        body = Text()
        body.append(f"-> {route.upper().replace('_', ' ')}", style=style)
        body.append(f"  [{', '.join(d['rule_ids'])}]", style="grey62")
        body.append(f"  {'; '.join(d['reasons'])}", style="white" if route != "skip" else "dim")
        self._row(rec, "gate", body)

    def _on_manager_decision(self, rec):
        d = rec["data"]
        nxt = d["next_agents"]
        body = Text()
        body.append(f"round {d['round']}: ", style="grey62")
        if nxt:
            body.append("wake ", style="white")
            for i, a in enumerate(nxt):
                body.append(a, style=f"bold {color_of(a)}")
                if i < len(nxt) - 1:
                    body.append(", ", style="white")
        else:
            body.append("no one else to wake", style="dim")
        if d.get("reason"):
            body.append(f"\n{d['reason']}", style="italic grey78")
        self._row(rec, "manager", body)

    def _on_critique_verdict(self, rec):
        d = rec["data"]
        lines = []
        for v in d.get("verdicts", []):
            st = {"keep": "green", "downgrade": "yellow", "drop": "bold red"}.get(v["verdict"], "white")
            t = Text()
            t.append(f"{v['verdict'].upper():9}", style=st)
            t.append(f"{v['agent']}/{v['type']}  ", style="white")
            t.append(v.get("reason", ""), style="grey62")
            lines.append(t)
        for c in d.get("contradictions", []):
            lines.append(Text(f"contradiction: {c}", style="bold red"))
        for r in d.get("requests", []):
            lines.append(Text(f"requests {r['to_agent']}: {r['reason']}", style="yellow"))
        self._row(rec, "critique", Panel(
            Group(Text(rec["message"], style="italic grey78"), *lines),
            title="critique", title_align="left", border_style=color_of("critique"),
            box=box.ROUNDED, padding=(0, 1)))

    def _on_tool_result(self, rec):
        d = rec["data"]
        self._row(rec, rec["agent"], Text.assemble(
            ("tool", "grey62"), (d["tool"], "bold"), (f"  ({d['elapsed_ms']}ms)  ", "grey50"), (clip(json.dumps(d["result"], default=str), 300), "grey70")
        ))

    def _on_life_inference(self, rec):
        d = rec["data"]
        body = [Text(rec["message"], style="italic grey78")]
        for h in d.get("hypotheses", []):
            conf = h["confidence_band"]
            t = Text()
            t.append(h["life_event"], style="bold")
            t.append(f"  {conf}", style=CONF_STYLE.get(conf, "white"))
            t.append(f"  <- {', '.join(h.get('supporting_agents', [])) or 'n/a'}", style="grey50")
            body += [t, Text(h["rationale"], style="grey70")]
        body.append(Text(f"urgency: {d.get('urgency')}  risks: {d.get('risk_themes')}  "
                         f"opportunities: {d.get('opportunity_themes')}", style="grey62"))
        self._row(rec, "life_inference", Panel(
            Group(*body), title="life inference", title_align="left",
            border_style=color_of("life_inference"), box=box.ROUNDED, padding=(0, 1)))

    def _on_llm_prompt(self, rec):
        d = rec["data"]
        n = 0 if self.verbosity >= 2 else 700
        parts = []
        if self.verbosity >= 2:
            parts.append(Text("SYSTEM\n", style="bold grey70"))
            parts.append(Text(clip(d.get("system", ""), n) + "\n\n", style="grey62"))
        parts.append(Text("USER\n", style="bold grey70"))
        parts.append(Text(clip(d.get("user", ""), n), style="grey62"))
        self._row(rec, rec["agent"], Panel(
            Group(*parts), title=f"LLM prompt -> {rec['agent']}  ({rec['message']})",
            title_align="left", border_style="grey37", box=box.ROUNDED, padding=(0, 1)))

    def _on_llm_response(self, rec):
        d = rec["data"]
        n = 0 if self.verbosity >= 2 else 900
        txt = json.dumps(d.get("response"), indent=2, default=str)
        self._row(rec, rec["agent"], Panel(
            Text(clip(txt, n), style="grey70"), title=f"LLM response <- {rec['agent']}  ({d.get('elapsed_s')}s)",
            title_align="left", border_style="grey37", box=box.ROUNDED, padding=(0, 1)))

    def _on_agent_start(self, rec):
        d = rec["data"]
        body = Text()
        body.append("thinking... ", style=color_of(rec["agent"]))
        body.append(f"{d.get('n_context_events', 'N/A')} event(s) in scope", style="grey70")
        body.append(f" | {d.get('n_prior_findings', 'N/A')} prior finding(s) in memory", style="grey50")
        self._row(rec, rec["agent"], body)

    def _on_agent_finding(self, rec):
        d = rec["data"]
        conf = d["confidence_band"]
        ev = ", ".join(d["evidence_event_ids"]) or "(none)"
        title = Text.assemble(("finding: ", "grey70"), (d["finding_type"], "bold"),
                              ("  confidence ", "grey62"), (conf, CONF_STYLE.get(conf, "white")))
        self._row(rec, rec["agent"], Panel(
            Group(Text(rec["message"]), Text(f"evidence: {ev}", style="grey50")),
            title=title, title_align="left", border_style=color_of(rec["agent"]),
            box=box.ROUNDED, padding=(0, 1)))

    def _on_handoff_request(self, rec):
        d = rec["data"]
        body = Text()
        body.append("asks ", style="white")
        body.append(d["to"], style=f"bold {color_of(d['to'])}")
        body.append(f" for help: {d['reason']}", style="white")
        self._row(rec, rec["agent"], body)

    def _on_handoff_review(self, rec):
        d = rec["data"]
        ok = d["verdict"] == "accepted"
        body = Text()
        body.append(f"handoff {d['from']} -> {d['to']}: ", style="white")
        body.append(d["verdict"], style="green" if ok else "bold red")
        self._row(rec, "manager", body)

    def _on_handoff_rejected(self, rec):
        self._row(rec, rec["agent"], Text(f"handoff rejected: {rec['message']}", style="bold red"))

    def _on_agent_done(self, rec):
        d = rec["data"]
        body = Text()
        body.append(f"done in {d['elapsed_s']}s ", style="green")
        body.append(f"({d['n_findings']} finding(s), {d['n_handoffs']} handoff(s)) ", style="grey62")
        body.append(rec["message"], style="italic grey78")
        if d.get("invalid_evidence_ids_dropped"):
            body.append(f"\ndropped invalid evidence ids: {d['invalid_evidence_ids_dropped']}", style="yellow")
        self._row(rec, rec["agent"], body)

    def _on_agent_error(self, rec):
        self._row(rec, rec["agent"], Text(f"ERROR {rec['message']}", style="bold red"))

    def _on_loop_guard(self, rec):
        self._row(rec, "manager", Text(f"LOOP GUARD: {rec['message']}", style="bold yellow"))

    def _on_finalize(self, rec):
        self._row(rec, "system", Text(f"finalize: {rec['message']}", style="grey78"))

    def _on_hitl_request(self, rec):
        d = rec.get("data") or {}
        self._row(
            rec,
            "hitl",
            Text(
                f"request={d.get('request_id')} action={d.get('action')} reason={d.get('reason')}",
                style="bold yellow",
            ),
            "bold yellow",
        )

    def _on_event_done(self, rec):
        d = rec["data"]
        body = Text()
        body.append(f"processed in {d['elapsed_s']}s", style="bold white")
        body.append(f" | gate={d.get('gate')} | rounds={d.get('manager_rounds')}", style="grey62")
        body.append(f" | agents run: {', '.join(d.get('routed_to') or []) or 'none'}", style="grey62")
        self._row(rec, "done", body, "bold white")
        self.console.print()

    def _on_error(self, rec):
        self._row(rec, "error", Panel(Text(rec["message"], style="red"), border_style="red", box=box.ROUNDED),
                  "bold red")

    # ------------------------------------------------------------ status bar
    def status_renderable(self):
        now = time.monotonic()
        chips = Text()
        for a in SPECIALISTS:
            st, c = self.status[a], color_of(a)
            if st["state"] == "running":
                chips.append(f" {a} ", style=f"black on {c}")
                chips.append(f" {now - st['since']:.0f}s", style=f"bold {c}")
            elif st["state"] == "done":
                chips.append(f" {a}", style=f"bold {c}")
                took = "" if st["last"] is None else " " + str(st["last"]) + "s"
                chips.append(" done" + took, style="grey62")
            elif st["state"] == "error":
                chips.append(f" {a} ERROR", style="bold red")
            else:
                chips.append(f" {a}", style="grey42")
                chips.append(" idle", style="grey42")
            chips.append("  |", style="grey30")
        c = self.counts
        stats = Text(
            f"live events {c['live']} | history {c['history']} | findings {c['findings']} | "
            f"handoffs {c['handoffs']} | LLM calls {c['llm calls']} | errors {c['errors']}   "
            f"[{self.connection}]", style="grey62")
        return Panel(Group(chips, stats), title="agents", title_align="left",
                     border_style="grey37", box=box.ROUNDED, padding=(0, 1))


# ------------------------------------------------------------------ sources
def run_live(mon: Monitor, url: str, interval: float, from_start: bool) -> None:
    sess = requests.Session()
    cursor, boot, tail_first = 0, None, not from_start
    while True:
        try:
            r = sess.get(f"{url}/trace",
                         params={"after": 10**12 if tail_first else cursor, "limit": POLL_LIMIT}, timeout=5)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            if mon.connection != "server unreachable":
                mon.connection = "server unreachable"
                mon.notice(f"cannot reach {url}/trace ({type(e).__name__}); retrying...", "red")
            time.sleep(1.0)
            continue

        if mon.connection != "connected":
            mon.connection = "connected"
            mon.notice(f"connected to {url}", "green")

        if boot is not None and data["boot_id"] != boot:
            mon.notice("server restarted -- trace cursor reset")
            cursor, boot, tail_first = 0, data["boot_id"], False
            continue
        boot = data["boot_id"]

        if tail_first:                      # default: start from "now", skip the backlog
            cursor, tail_first = data["last_id"], False
            continue

        if data["events"] and cursor >= 1 and data["oldest_id"] > cursor + 1:
            mon.notice(f"monitor fell behind: {data['oldest_id'] - cursor - 1} event(s) dropped from buffer")

        for rec in data["events"]:
            mon.handle(rec)
            cursor = rec["id"]
        if len(data["events"]) < POLL_LIMIT:
            time.sleep(interval)


def run_replay(mon: Monitor, path: Path, speed: float) -> None:
    prev = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if speed > 0 and prev is not None:
                try:
                    gap = (datetime.fromisoformat(rec["ts"]) - datetime.fromisoformat(prev)).total_seconds()
                    time.sleep(min(max(gap, 0) / speed, 3.0))
                except Exception:
                    pass
            prev = rec["ts"]
            mon.handle(rec)


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Live view of the Customer360 agentic layer.")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="server base url")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v: LLM prompts/responses; -vv: full prompts + system prompts + memory updates")
    ap.add_argument("--from-start", action="store_true", help="show the server's whole trace buffer, not just new events")
    ap.add_argument("--history", action="store_true", help="also show history-seed ingestion lines")
    ap.add_argument("--customer", help="only show this customer_id")
    ap.add_argument("--agent", nargs="+", help="only show these agents (e.g. transaction compliance manager)")
    ap.add_argument("--interval", type=float, default=0.25, help="poll interval in seconds")
    ap.add_argument("--no-status", action="store_true", help="disable the pinned agent status bar")
    ap.add_argument("--replay", type=Path, help="replay a saved trace.jsonl instead of polling the server")
    ap.add_argument("--speed", type=float, default=0.0, help="replay speed factor (0 = instant)")
    args = ap.parse_args()

    console = Console()
    mon = Monitor(console, verbosity=args.verbose, show_history=args.history,
                  agents=set(args.agent or []), customer=args.customer)

    console.print(Text("Customer360 | Agent Monitor", style="bold cyan"))
    src = f"replaying {args.replay}" if args.replay else f"polling {args.url}/trace"
    console.print(Text(f"{src} | verbosity {args.verbose} | Ctrl+C to quit\n", style="grey62"))

    use_status = console.is_terminal and not args.no_status
    live = Live(get_renderable=mon.status_renderable, console=console,
                refresh_per_second=6, transient=False) if use_status else None
    try:
        if live:
            live.start()
        if args.replay:
            run_replay(mon, args.replay, args.speed)
        else:
            run_live(mon, args.url.rstrip("/"), args.interval, args.from_start)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    finally:
        if live:
            live.stop()
    console.print(Text("\nmonitor stopped.", style="grey62"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
