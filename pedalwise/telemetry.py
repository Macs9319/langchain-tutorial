"""What the agent did on each run, and what customers thought of it (Tutorial 22).

Logs (Tutorial 21) say what the server did. They can't answer the questions
you actually have about an LLM app: Which answers do customers dislike? Is
it getting slower? How many tokens does a chat cost? Which tools are used,
and how often do they fail?

So every chat turn becomes a *run* with an id, and this module saves one
summary row per run, plus the thumbs up/down a customer gives it. That
works with no account anywhere. If LangSmith tracing is switched on, the
same run id names the full trace there (every prompt, tool call and
timing), and feedback is sent to it too, so you can jump from a thumbs-down
here to exactly what the model saw.

Privacy: runs keep the question and the answer, because a thumbs-down is
useless without them. They're deleted after TELEMETRY_RETENTION_DAYS.
"""

import json
import time
from collections import Counter
from typing import Any

import aiosqlite
from langchain_core.callbacks import BaseCallbackHandler

MAX_TEXT = 2000  # characters of question/answer kept per run


class RunRecorder(BaseCallbackHandler):
    """Counts model calls, tokens and tool calls during one run.

    Pass it in the run's config: {"callbacks": [RunRecorder()]}. LangChain
    calls these methods as the agent works, including inside middleware
    (summarization, retries and fallbacks all count).
    """

    run_inline = True  # just counting: no need to run in a worker thread

    def __init__(self):
        self.started = time.perf_counter()
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tools: list[str] = []
        self.tool_errors = 0

    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.model_calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def on_tool_start(self, serialized, input_str, **kwargs: Any) -> None:
        self.tools.append(kwargs.get("name") or (serialized or {}).get("name", "unknown"))

    def on_tool_error(self, error, **kwargs: Any) -> None:
        self.tool_errors += 1

    @property
    def latency_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


class Telemetry:
    """Runs and feedback in SQLite. Async for the same reason as ApprovalQueue."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    @classmethod
    async def open(cls, db_path: str) -> "Telemetry":
        db = await aiosqlite.connect(db_path, timeout=10)
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id        TEXT PRIMARY KEY,
                thread_id     TEXT NOT NULL,
                customer      TEXT NOT NULL,
                kind          TEXT NOT NULL,     -- chat or staff_decision
                outcome       TEXT NOT NULL,     -- ok, approval, error, timeout
                latency_ms    INTEGER NOT NULL,
                model_calls   INTEGER NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                tools         TEXT NOT NULL,     -- JSON list of tool names
                tool_errors   INTEGER NOT NULL,
                question      TEXT,
                answer        TEXT,
                created       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS feedback (
                run_id  TEXT PRIMARY KEY REFERENCES runs(run_id),
                score   INTEGER NOT NULL,        -- 1 thumbs up, -1 thumbs down
                comment TEXT,
                created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS runs_created ON runs(created);
        """)
        await db.commit()
        return cls(db)

    async def record(self, run_id: str, thread_id: str, customer: str, kind: str, outcome: str,
                     recorder: RunRecorder, question: str | None, answer: str | None) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO runs (run_id, thread_id, customer, kind, outcome, latency_ms,"
            " model_calls, input_tokens, output_tokens, tools, tool_errors, question, answer)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, thread_id, customer, kind, outcome, recorder.latency_ms,
             recorder.model_calls, recorder.input_tokens, recorder.output_tokens,
             json.dumps(recorder.tools), recorder.tool_errors,
             (question or "")[:MAX_TEXT] or None, (answer or "")[:MAX_TEXT] or None))
        await self._db.commit()

    async def owner(self, run_id: str) -> str | None:
        async with self._db.execute("SELECT customer FROM runs WHERE run_id = ?",
                                    (run_id,)) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    async def add_feedback(self, run_id: str, score: int, comment: str | None) -> None:
        # INSERT OR REPLACE: a customer who changes their mind replaces their vote.
        await self._db.execute(
            "INSERT OR REPLACE INTO feedback (run_id, score, comment) VALUES (?, ?, ?)",
            (run_id, score, comment))
        await self._db.commit()

    async def delete_older_than(self, days: int) -> int:
        async with self._db.execute(
                "DELETE FROM feedback WHERE run_id IN (SELECT run_id FROM runs"
                " WHERE created < datetime('now', ?))", (f"-{days} days",)):
            pass
        async with self._db.execute("DELETE FROM runs WHERE created < datetime('now', ?)",
                                    (f"-{days} days",)) as cursor:
            deleted = cursor.rowcount
        await self._db.commit()
        return deleted

    async def negative_feedback(self, limit: int = 100) -> list[dict]:
        """Thumbs-down runs, newest first: the raw material for new eval cases."""
        async with self._db.execute(
                "SELECT r.run_id, r.thread_id, r.question, r.answer, f.comment, f.created"
                " FROM feedback f JOIN runs r USING (run_id)"
                " WHERE f.score < 0 ORDER BY f.created DESC LIMIT ?", (limit,)) as cursor:
            keys = ("run_id", "thread_id", "question", "answer", "comment", "created")
            return [dict(zip(keys, row)) async for row in cursor]

    async def metrics(self, hours: int = 24) -> dict:
        """A summary of the last `hours`: volume, errors, speed, cost, tools, feedback."""
        since = (f"-{hours} hours",)
        async with self._db.execute(
                "SELECT outcome, latency_ms, model_calls, input_tokens, output_tokens, tools,"
                " tool_errors FROM runs WHERE created >= datetime('now', ?)", since) as cursor:
            rows = [row async for row in cursor]
        async with self._db.execute(
                "SELECT f.score, COUNT(*) FROM feedback f JOIN runs r USING (run_id)"
                " WHERE r.created >= datetime('now', ?) GROUP BY f.score", since) as cursor:
            votes = {score: count async for score, count in cursor}

        outcomes = Counter(row[0] for row in rows)
        latencies = [row[1] for row in rows]
        tools = Counter(name for row in rows for name in json.loads(row[5]))
        failed = outcomes["error"] + outcomes["timeout"]
        rated = votes.get(1, 0) + votes.get(-1, 0)
        return {
            "hours": hours,
            "runs": len(rows),
            "outcomes": dict(outcomes),
            "error_rate": round(failed / len(rows), 3) if rows else 0.0,
            "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
            "model_calls_per_run": round(sum(r[2] for r in rows) / len(rows), 2) if rows else 0.0,
            "tokens": {"input": sum(r[3] for r in rows), "output": sum(r[4] for r in rows)},
            "tools": dict(tools.most_common()),
            "tool_errors": sum(r[6] for r in rows),
            "feedback": {"up": votes.get(1, 0), "down": votes.get(-1, 0),
                         "rated_share": round(rated / len(rows), 3) if rows else 0.0},
        }

    async def close(self) -> None:
        await self._db.close()
