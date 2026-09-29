#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from textwrap import shorten

import requests
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _post(base_url: str, path: str, payload: dict) -> dict:
    r = requests.post(f"{base_url}{path}", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def _get(base_url: str, path: str, **params) -> dict:
    r = requests.get(f"{base_url}{path}", params=params or None, timeout=15)
    r.raise_for_status()
    return r.json()


def _parse_modify(parts: list[str]) -> tuple[str, str, str | None, str | None]:
    if len(parts) < 3:
        raise ValueError("usage: /modify <request_id> <action> [--subtype X] [--notes \"...\"]")
    request_id = parts[1]
    action = parts[2]
    subtype = None
    notes = None
    i = 3
    while i < len(parts):
        token = parts[i]
        if token == "--subtype":
            if i + 1 >= len(parts):
                raise ValueError("--subtype requires a value")
            subtype = parts[i + 1]
            i += 2
            continue
        if token == "--notes":
            if i + 1 >= len(parts):
                raise ValueError("--notes requires a value")
            notes = parts[i + 1]
            i += 2
            continue
        raise ValueError(f"unknown flag: {token}")
    return request_id, action, subtype, notes


class HitlCli:
    def __init__(self, base_url: str, reviewer: str, default_limit: int):
        self.base_url = base_url
        self.reviewer = reviewer
        self.default_limit = default_limit
        self.console = Console()

    def print_banner(self) -> None:
        title = Text("Customer360 HITL Console", style="bold cyan")
        subtitle = Text(f"{self.base_url}  •  reviewer={self.reviewer}", style="grey70")
        self.console.print(Panel(Text.assemble(title, "\n", subtitle), border_style="cyan", box=box.ROUNDED))
        self.console.print(Text("Slash commands only. Press Enter on empty input to stay idle.", style="grey62"))
        self.console.print(Text("Try /hitl to view pending requests, or /help for commands.\n", style="grey62"))

    def print_help(self) -> None:
        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold")
        table.add_column("Description")
        table.add_row("/hitl [limit]", "List pending HITL requests")
        table.add_row("/approve <request_id> [notes]", "Approve request")
        table.add_row("/reject <request_id> [notes]", "Reject request (sets no_action)")
        table.add_row("/modify <request_id> <action> [--subtype X] [--notes \"...\"]", "Override action and finalize")
        table.add_row("/ask <request_id> [question]", "Ask for explanation (stateless)")
        table.add_row("/trace <request_id> [limit]", "Show event trace backing this request")
        table.add_row("/clear", "Clear screen")
        table.add_row("/help", "Show command reference")
        table.add_row("/exit", "Exit console")
        self.console.print(table)

    def show_pending(self, limit: int) -> None:
        data = _get(self.base_url, "/hitl/pending", limit=limit)
        rows = data.get("pending") or []
        if not rows:
            self.console.print(Panel("No pending HITL requests.", border_style="green", box=box.ROUNDED))
            return

        table = Table(title=f"Pending HITL Requests ({len(rows)})", box=box.SIMPLE_HEAVY, header_style="bold yellow")
        table.add_column("Request ID", style="cyan")
        table.add_column("Customer")
        table.add_column("Event")
        table.add_column("Action")
        table.add_column("Confidence")
        table.add_column("Reason")
        for row in rows:
            decision = row.get("decision") or {}
            table.add_row(
                str(row.get("request_id")),
                str(row.get("customer_id")),
                str(row.get("event_id")),
                str(decision.get("action") or "-"),
                str(decision.get("confidence_band") or "-"),
                shorten(str(row.get("reason") or "-"), width=80, placeholder="..."),
            )
        self.console.print(table)

    def show_decision_result(self, out: dict, heading: str, color: str) -> None:
        decision = out.get("decision") or {}
        body = Text()
        body.append(f"request: {out.get('request_id')}\n", style="grey70")
        body.append(f"status: {out.get('status')}\n", style=f"bold {color}")
        body.append(f"final action: {decision.get('action')} ({decision.get('hitl_status')})", style="white")
        self.console.print(Panel(body, title=heading, border_style=color, box=box.ROUNDED))

    def show_answer(self, out: dict) -> None:
        mode = out.get("mode") or "unknown"
        self.console.print(
            Panel(
                out.get("answer") or "",
                title=f"Explanation • {out.get('request_id')} • mode={mode}",
                border_style="blue",
                box=box.ROUNDED,
            )
        )
        citations = out.get("citations") or []
        if citations:
            table = Table(title="Citations", box=box.SIMPLE_HEAVY, header_style="bold magenta")
            table.add_column("Type", style="bold")
            table.add_column("Details")
            for c in citations:
                ctype = str(c.get("type") or "ref")
                detail = ", ".join(f"{k}={v}" for k, v in c.items() if k != "type")
                table.add_row(ctype, detail)
            self.console.print(table)

    def show_trace(self, request_id: str, limit: int) -> None:
        out = _get(self.base_url, f"/hitl/trace/{request_id}", limit=limit)
        rows = out.get("trace") or []
        table = Table(title=f"Trace • {request_id} ({len(rows)} rows)", box=box.SIMPLE_HEAVY, header_style="bold green")
        table.add_column("ID", justify="right")
        table.add_column("Kind")
        table.add_column("Agent")
        table.add_column("Message")
        for rec in rows:
            table.add_row(
                str(rec.get("id", "")),
                str(rec.get("kind", "")),
                str(rec.get("agent", "")),
                shorten(str(rec.get("message", "")), width=90, placeholder="..."),
            )
        self.console.print(table)

    def run(self) -> int:
        self.print_banner()
        while True:
            try:
                line = self.console.input("[bold cyan]hitl[/bold cyan] [grey62]>[/grey62] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return 0

            if not line:
                continue
            if not line.startswith("/"):
                self.console.print("Use slash commands (for example: /hitl or /ask <request_id> why?).", style="yellow")
                continue

            try:
                parts = shlex.split(line[1:])
            except ValueError as e:
                self.console.print(f"Parse error: {e}", style="bold red")
                continue
            if not parts:
                continue

            cmd = parts[0].lower()
            try:
                if cmd in {"exit", "quit"}:
                    return 0
                if cmd == "help":
                    self.print_help()
                    continue
                if cmd == "clear":
                    self.console.clear()
                    self.print_banner()
                    continue
                if cmd == "hitl":
                    limit = int(parts[1]) if len(parts) > 1 else self.default_limit
                    self.show_pending(limit=limit)
                    continue
                if cmd == "approve":
                    if len(parts) < 2:
                        self.console.print("usage: /approve <request_id> [notes]", style="yellow")
                        continue
                    request_id = parts[1]
                    notes = " ".join(parts[2:]).strip() or None
                    out = _post(
                        self.base_url,
                        "/hitl/approve",
                        {"request_id": request_id, "reviewer": self.reviewer, "human_notes": notes},
                    )
                    self.show_decision_result(out, "Approved", "green")
                    continue
                if cmd == "reject":
                    if len(parts) < 2:
                        self.console.print("usage: /reject <request_id> [notes]", style="yellow")
                        continue
                    request_id = parts[1]
                    notes = " ".join(parts[2:]).strip() or None
                    out = _post(
                        self.base_url,
                        "/hitl/reject",
                        {"request_id": request_id, "reviewer": self.reviewer, "human_notes": notes},
                    )
                    self.show_decision_result(out, "Rejected", "red")
                    continue
                if cmd == "modify":
                    request_id, action, subtype, notes = _parse_modify(parts)
                    out = _post(
                        self.base_url,
                        "/hitl/modify",
                        {
                            "request_id": request_id,
                            "reviewer": self.reviewer,
                            "human_notes": notes,
                            "modified_action": action,
                            "modified_action_subtype": subtype,
                        },
                    )
                    self.show_decision_result(out, "Modified", "cyan")
                    continue
                if cmd == "ask":
                    if len(parts) < 2:
                        self.console.print("usage: /ask <request_id> [question]", style="yellow")
                        continue
                    request_id = parts[1]
                    question = " ".join(parts[2:]).strip() or "why is this pending approval?"
                    out = _post(self.base_url, "/hitl/ask", {"request_id": request_id, "question": question})
                    self.show_answer(out)
                    continue
                if cmd == "trace":
                    if len(parts) < 2:
                        self.console.print("usage: /trace <request_id> [limit]", style="yellow")
                        continue
                    request_id = parts[1]
                    limit = int(parts[2]) if len(parts) > 2 else 30
                    self.show_trace(request_id, limit=limit)
                    continue

                self.console.print(f"Unknown command: /{cmd}. Type /help.", style="yellow")
            except requests.HTTPError as e:
                body = e.response.text if e.response is not None else str(e)
                self.console.print(Panel(body, title="Request failed", border_style="red", box=box.ROUNDED))
            except Exception as e:
                self.console.print(Panel(str(e), title="Error", border_style="red", box=box.ROUNDED))


def main() -> int:
    ap = argparse.ArgumentParser(description="Customer360 HITL CLI")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="server base url")
    ap.add_argument("--reviewer", default="human", help="reviewer identity stored in audit logs")
    ap.add_argument("--limit", type=int, default=20, help="default row limit for /hitl")
    args = ap.parse_args()
    return HitlCli(args.url.rstrip("/"), args.reviewer, default_limit=args.limit).run()


if __name__ == "__main__":
    sys.exit(main())
