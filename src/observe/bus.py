"""
Trace bus: the ONE place the agentic layer writes "what is happening" to.

    graph nodes / agents --emit()--> ring buffer  ---> GET /trace?after=N  <-- monitor CLI (live)
                                 `-> writer thread -> SQLite `trace_events` (system of record)
                                 `-> traces/<run_id>.jsonl (raw copy, for `monitor --replay`)

PS requirement this serves: "for any final action it should be possible to pull up exactly which
prompts were sent, which tools were called and what each returned". Every record carries
(run_id, customer_id, event_id), so a decision row (which stores its trigger event_id)
joins to its full trace with:  SELECT * FROM trace_events WHERE run_id=? AND event_id=? ORDER BY id

Rules
- emit() must NEVER raise or block the pipeline: the DB write happens on a background thread.
- One run per server start (RUN_ID). Old runs stay in the DB so runs can be compared / scored.
- Env: C360_DB (sqlite path, default src/server/server.db), TRACE_DIR (jsonl dir, "" disables).
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import sqlite3
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_BUFFER = 5000
RUN_ID = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
BOOT_ID = RUN_ID                       # monitor uses this to notice a server restart
DB_PATH = Path(os.getenv("C360_DB") or Path(__file__).resolve().parents[1] / "server" / "server.db")
_TRACE_DIR = os.getenv("TRACE_DIR", "traces")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_events (
    run_id      TEXT NOT NULL,
    id          INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    agent       TEXT NOT NULL,
    customer_id TEXT,
    event_id    TEXT,
    level       TEXT NOT NULL,
    message     TEXT,
    data        TEXT,
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS ix_trace_event ON trace_events(run_id, event_id);
CREATE INDEX IF NOT EXISTS ix_trace_kind  ON trace_events(run_id, kind);
"""

_lock = threading.Lock()
_buffer: deque[dict] = deque(maxlen=MAX_BUFFER)
_next_id = 1

_file = None
if _TRACE_DIR:
    try:
        Path(_TRACE_DIR).mkdir(parents=True, exist_ok=True)
        _file = open(Path(_TRACE_DIR) / f"{RUN_ID}.jsonl", "w", encoding="utf-8", buffering=1)
    except OSError:
        _file = None

# ---------------------------------------------------------------- sqlite sink
_q: "queue.SimpleQueue" = queue.SimpleQueue()
_warned = False


def _writer_loop() -> None:
    global _warned
    conn = None
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
    except Exception as e:                       # noqa: BLE001
        print(f"[observe] trace DB unavailable ({e!r}); continuing without SQL persistence", file=sys.stderr)

    stop = False
    while not stop:
        batch = [_q.get()]
        try:
            while len(batch) < 200:
                batch.append(_q.get(timeout=0.15))
        except queue.Empty:
            pass
        rows, waiters = [], []
        for item in batch:
            if item is None:
                stop = True
            elif item[0] == "flush":
                waiters.append(item[1])
            else:
                rows.append(item[1])
        if rows and conn is not None:
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO trace_events VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                conn.commit()
            except Exception as e:               # noqa: BLE001
                if not _warned:
                    _warned = True
                    print(f"[observe] trace DB write failed: {e!r}", file=sys.stderr)
        for w in waiters:
            w.set()
    if conn is not None:
        conn.close()


_writer = threading.Thread(target=_writer_loop, name="trace-writer", daemon=True)
_writer.start()


def flush(timeout: float = 3.0) -> bool:
    """Block until everything emitted so far is committed to SQLite (used before reads/at exit)."""
    ev = threading.Event()
    _q.put(("flush", ev))
    return ev.wait(timeout)


@atexit.register
def _shutdown() -> None:
    flush(3.0)
    _q.put(None)
    _writer.join(timeout=3.0)


# ------------------------------------------------------------------ public api
def _clip(value: Any, limit: int = 20000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [+{len(value) - limit} chars]"
    return value


def emit(
    kind: str,
    agent: str = "system",
    message: str = "",
    *,
    customer_id: str | None = None,
    event_id: str | None = None,
    level: str = "info",          # "info" | "debug" (debug = hidden by default in the monitor)
    data: dict | None = None,
) -> int | None:
    """Record one trace event. Returns its id (or None if tracing failed)."""
    global _next_id
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "kind": kind,
            "agent": agent,
            "customer_id": customer_id,
            "event_id": event_id,
            "level": level,
            "message": _clip(message, 2000),
            "data": {k: _clip(v) for k, v in (data or {}).items()},
        }
        with _lock:
            rec["id"] = _next_id
            _next_id += 1
            _buffer.append(rec)
            if _file:
                _file.write(json.dumps(rec, default=str) + "\n")
            _q.put(("row", (RUN_ID, rec["id"], rec["ts"], kind, agent, customer_id, event_id,
                            level, rec["message"], json.dumps(rec["data"], default=str))))
        return rec["id"]
    except Exception:
        return None


def read_since(after: int = 0, limit: int = 500) -> dict:
    """Live tail for the monitor: events with id > after (oldest first)."""
    with _lock:
        items = [r for r in _buffer if r["id"] > after][:limit]
        oldest = _buffer[0]["id"] if _buffer else 0
        last = _next_id - 1
    return {"boot_id": BOOT_ID, "events": items, "last_id": last, "oldest_id": oldest}


def trace_for_event(event_id: str, run_id: str | None = None) -> list[dict]:
    """Full stored trace (prompts, responses, findings, handoffs...) behind one event's processing."""
    flush()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM trace_events WHERE run_id=? AND event_id=? ORDER BY id",
            (run_id or RUN_ID, event_id)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = json.loads(d["data"] or "{}")
        out.append(d)
    return out


def clear() -> None:
    with _lock:
        _buffer.clear()
