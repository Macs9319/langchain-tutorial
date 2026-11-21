"""
Tutorial 22 — Observability: seeing what your deployed agent does
=================================================================

Tutorial 21 deployed the app; its logs say a request came in and took two
seconds. They don't say whether the answer was any good, why it was slow,
or what it cost. For an LLM app those are the questions that matter, and
the answers live in the runs: what the model was asked, which tools it
called, how many tokens it used, and whether the customer was happy.

This tutorial adds that to the Pedalwise app, in three parts:

  1. Runs. Every chat turn gets a run id. A callback handler counts model
     calls, tokens and tool calls while the agent works, and one summary
     row per run is saved (pedalwise/telemetry.py).
  2. Feedback. Each answer in the chat page gets 👍 / 👎, and a thumbs-down
     asks what was wrong. The vote is stored against that run id.
  3. The loop back to evaluation. Thumbs-down answers, with the customer's
     comment, are exported as draft test cases for Tutorial 10. Every
     complaint becomes a test, so the same mistake can't quietly return.

Staff see it all on /staff: run counts, error rate, typical and slow reply
times (p50 / p95), tokens, tool usage and the latest thumbs-down answers.

What you'll learn:
  1. Run ids, tags and metadata on a LangChain run config
  2. Writing a callback handler to measure what an agent does
  3. Collecting user feedback tied to the exact answer it's about
  4. Which numbers to watch: p95 latency, error rate, tokens per run
  5. Turning production complaints into evaluation cases

Optional: LangSmith for full traces
  The summary rows above work with no account. For the full trace of a run
  (every prompt, retrieved chunk, tool input and output, with timings), add
  to .env (free account at smith.langchain.com):
      LANGSMITH_TRACING=true
      LANGSMITH_API_KEY=lsv2_...
      LANGSMITH_PROJECT=pedalwise
  No code changes: the run id the app chooses becomes the trace id, the
  tags ("production", "support-chat") and metadata (customer) become
  filters, and feedback is sent to the trace too. To see why a thumbs-down
  answer went wrong, search the project for its run id.
  Mind that traces contain customers' messages and order details: that's
  data leaving your servers, which your privacy policy has to allow.

Setup:
  Same as Tutorial 20 (OPENAI_API_KEY in .env).

Run:
  uv run 22_observability.py                  # demo: 3 questions, votes, report (~6 model calls)
  uv run 22_observability.py --report         # report from the app's database (no API calls)
  uv run 22_observability.py --report --hours 168
  uv run 22_observability.py --export-evals evals_from_feedback.jsonl

  Or use the real app: uv run 20_capstone.py, chat and vote at
  http://127.0.0.1:8001, and watch the numbers at http://127.0.0.1:8001/staff.
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from langsmith.utils import tracing_is_enabled

from pedalwise.config import Settings
from pedalwise.server import create_app
from pedalwise.telemetry import Telemetry


# ---------------------------------------------------------------------------
# Reading the numbers
# ---------------------------------------------------------------------------
def print_report(m: dict) -> None:
    """What to look at, in order of how much it tells you."""
    print(f"\nLast {m['hours']} hours: {m['runs']} run{'s' * (m['runs'] != 1)} {m['outcomes']}")
    if not m["runs"]:
        return
    fb = m["feedback"]
    # Feedback first: it's the only number that measures answer quality.
    print(f"  feedback        👍 {fb['up']}  👎 {fb['down']}  "
          f"({fb['rated_share']:.0%} of runs rated; most people never vote)")
    # Errors and timeouts: each one is a customer who got no answer.
    print(f"  error rate      {m['error_rate']:.1%}")
    # p95, not the average: the slowest 1 in 20 replies is what people complain about.
    print(f"  latency         p50 {m['latency_ms']['p50']} ms, p95 {m['latency_ms']['p95']} ms")
    # Model calls per run: if this creeps up, the agent is looping or flailing.
    print(f"  model calls     {m['model_calls_per_run']} per run")
    # Tokens are the bill. Multiply by your model's price per million tokens.
    print(f"  tokens          {m['tokens']['input']:,} in, {m['tokens']['output']:,} out")
    print(f"  tools           {m['tools'] or 'none'}"
          + (f", {m['tool_errors']} errors" if m["tool_errors"] else ""))


async def report(db_path: str, hours: int) -> None:
    if not Path(db_path).exists():
        print(f"No database at {db_path} yet. Run the app (20 or 21) and chat first.")
        return
    telemetry = await Telemetry.open(db_path)
    try:
        print_report(await telemetry.metrics(hours))
        negative = await telemetry.negative_feedback(limit=5)
        if negative:
            print("\nLatest thumbs-down:")
            for item in negative:
                answer = " ".join((item["answer"] or "").split())  # one line
                print(f"  Q: {item['question']}\n  A: {answer[:120]}")
                print(f"     customer said: {item['comment'] or '(no comment)'}\n")
    finally:
        await telemetry.close()


# ---------------------------------------------------------------------------
# Closing the loop: complaints become test cases
# ---------------------------------------------------------------------------
async def export_evals(db_path: str, out: Path) -> None:
    """Write thumbs-down runs as draft cases in Tutorial 10's format.

    A person still has to fill in "reference" (what a good answer says) and
    "must_include": a complaint tells you an answer was wrong, not what the
    right one is. Then copy the case into TEST_SET in 10_evaluation.py.
    """
    telemetry = await Telemetry.open(db_path)
    try:
        negative = await telemetry.negative_feedback(limit=1000)
    finally:
        await telemetry.close()
    written = 0
    with out.open("w", encoding="utf-8") as file:
        for item in negative:
            if not item["question"]:
                continue  # staff decisions have no customer question
            written += 1
            file.write(json.dumps({
                "question": item["question"],
                "reference": "",          # TODO: what a correct answer says
                "must_include": [],       # TODO: key facts, as in Tutorial 10
                "bad_answer": item["answer"],
                "customer_comment": item["comment"],
                "run_id": item["run_id"],  # find the full trace in LangSmith
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {written} draft case(s) to {out}. Fill in 'reference' and "
          "'must_include', then add them to TEST_SET in 10_evaluation.py.")


# ---------------------------------------------------------------------------
# Demo: the real app in-process, with a throwaway database
# ---------------------------------------------------------------------------
def read_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields.get("event"), json.loads(fields.get("data", "{}"))))
    return events


def demo() -> None:
    print("LangSmith tracing is", "ON: open your project to see each trace."
          if tracing_is_enabled() else "off (optional; see the top of this file).")
    with tempfile.TemporaryDirectory() as tmp:
        # A fresh database so the report shows only this demo. The search
        # index is the shared one, so documents aren't embedded again.
        settings = Settings(db_path=str(Path(tmp) / "demo.sqlite"))
        staff = {"X-Staff-Token": settings.staff_token}
        with TestClient(create_app(settings)) as client:
            runs = {}
            for question in ["What time does the Porto shop open on Saturday?",
                             "Do you sell skateboards?",
                             "What are my orders?"]:
                print(f"\nCustomer: {question}")
                response = client.post("/api/chat", json={
                    "message": question, "customer_email": "sam@example.com"})
                events = read_events(response.text)
                runs[question] = next(d["run_id"] for e, d in events if e == "run")
                answer = "".join(d["text"] for e, d in events if e == "token")
                print(f"Assistant: {answer}")
                print(f"  run id: {runs[question]}")

            # The customer votes, the way the 👍 / 👎 buttons do.
            vote = lambda q, score, comment=None: client.post("/api/feedback", json={
                "run_id": runs[q], "customer_email": "sam@example.com",
                "score": score, "comment": comment})
            vote("What time does the Porto shop open on Saturday?", 1)
            vote("Do you sell skateboards?", -1, "I just wanted a yes or no")
            print("\nVotes: 👍 on the opening hours, 👎 on skateboards (with a comment).")

            # What staff see on /staff.
            m = client.get("/api/staff/metrics", headers=staff).json()
            print_report(m)
            print("\nThumbs-down, ready to become a test case:")
            for item in m["negative_feedback"]:
                print(f"  Q: {item['question']}\n  customer said: {item['comment']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observability for the Pedalwise app.")
    parser.add_argument("--report", action="store_true", help="metrics from the app's database")
    parser.add_argument("--hours", type=int, default=24, help="with --report: time window")
    parser.add_argument("--export-evals", metavar="FILE", type=Path,
                        help="write thumbs-down runs as draft eval cases (JSON lines)")
    parser.add_argument("--db", default=None, help="database path (default: the app's)")
    args = parser.parse_args()
    db_path = args.db or Settings().db_path
    if args.report:
        asyncio.run(report(db_path, args.hours))
    elif args.export_evals:
        asyncio.run(export_evals(db_path, args.export_evals))
    else:
        demo()
