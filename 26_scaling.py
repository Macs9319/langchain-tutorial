"""
Tutorial 26 — Scaling out: several copies of the app
====================================================

One copy of the app (Tutorial 21) has limits: it serves as many chats as
one machine can, and when it restarts, everyone waits. Running several
copies behind a load balancer fixes both, but only if any copy can handle
any request. That means no copy may keep anything important to itself.

Going through the capstone for state kept in one process found:

    conversations       SQLite file on one machine      -> Postgres checkpointer
    approval queue      SQLite file on one machine      -> Postgres table
    telemetry           SQLite file on one machine      -> Postgres table
    refunds paid        a Python list in memory!        -> Postgres table
    search index        files built from data/          -> fine: a cache each copy
                                                           can rebuild on its own

The refund list was a real bug, not only a scaling problem: since Tutorial
20 a restart emptied it, so an order refunded before a deploy looked
refundable again. With two copies it was worse: each had its own list, so
both could pay. It's now a table whose primary key is the order id, so the
database refuses a second refund for the same order even when two copies
try at the same instant (pedalwise/refunds.py).

What you'll learn:
  1. Find the state: every variable that outlives one request is suspect
  2. Move it to a shared database, through a small layer that also keeps
     SQLite working for local runs (pedalwise/database.py)
  3. Let the database enforce "only once" (a primary key), not Python
  4. Start safely: several copies creating tables at once need a lock
  5. Put a load balancer in front (Caddy), and why no "sticky sessions"
     are needed once state is shared
  6. Check it from the outside: requests spread, conversations move between
     copies, and a double approval pays once

Setup:
  Docker, and in .env: an API key and STAFF_TOKEN (the same for every copy).

Run:
  docker compose -f compose.scale.yaml up --build -d
  uv run 26_scaling.py --check http://127.0.0.1:8080     # ~4 model calls
  docker compose -f compose.scale.yaml up -d --scale pedalwise=3
  docker compose -f compose.scale.yaml stop pedalwise    # then start it again:
                                                         # nothing is lost
Or run the tests on Postgres:
  TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/db uv run pytest
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()


def call(base: str, path: str, body: dict | None = None, staff: bool = False,
         timeout: float = 120) -> tuple[int, str, str]:
    """(status, body, which copy served it)."""
    headers = {"content-type": "application/json"}
    if staff:
        headers["X-Staff-Token"] = os.getenv("STAFF_TOKEN", "")
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(), response.headers["X-Served-By"]
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), error.headers.get("X-Served-By", "?")


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields.get("event"), json.loads(fields.get("data", "{}"))))
    return events


def chat(base, message, customer, thread_id=None):
    status, text, served_by = call(base, "/api/chat", {
        "message": message, "customer_email": customer, "thread_id": thread_id})
    events = sse_events(text) if status == 200 else []
    thread = next((d["thread_id"] for e, d in events if e == "thread"), thread_id)
    reply = "".join(d["text"] for e, d in events if e == "token")
    pending = any(e == "approval_pending" for e, _ in events)
    return thread, reply, pending, served_by


# ---------------------------------------------------------------------------
# Check 1: requests spread across copies
# ---------------------------------------------------------------------------
def check_spread(base: str) -> bool:
    with ThreadPoolExecutor(10) as pool:
        served = list(pool.map(lambda _: call(base, "/healthz")[2], range(30)))
    counts = Counter(served)
    print(f"30 requests were served by {len(counts)} copies: {dict(counts)}")
    return len(counts) > 1


# ---------------------------------------------------------------------------
# Check 2: a conversation started on one copy continues on another
# ---------------------------------------------------------------------------
def check_shared_conversation(base: str) -> bool:
    thread, reply, _, first = chat(base, "What are my orders? One line.", "sam@example.com")
    print(f"chat on {first}: {reply[:70]!r}")
    # Keep asking for the conversation until a different copy answers.
    for _ in range(20):
        status, text, other = call(base, f"/api/threads/{thread}?customer_email=sam@example.com")
        if other != first:
            saved = json.loads(text)["messages"] if status == 200 else []
            print(f"read back on {other}: {len(saved)} messages")
            return status == 200 and len(saved) >= 2
    print("every request reached the same copy; is there only one?")
    return False


# ---------------------------------------------------------------------------
# Check 3: two staff members approve the same refund at the same moment
# ---------------------------------------------------------------------------
def check_double_approval(base: str) -> bool | None:
    status, text, _ = call(base, "/api/staff/refunds", staff=True)
    if status == 401:
        print("STAFF_TOKEN in .env doesn't match the app's. Skipping.")
        return False
    paid = {r["order_id"] for r in json.loads(text)}
    order, customer = next(((o, c) for o, c in [("A1001", "sam@example.com"),
                                                 ("A1002", "dana@example.com")]
                            if o not in paid), (None, None))
    if order is None:
        print("A1001 and A1002 are already refunded in this database. For a fresh one:\n"
              "  docker compose -f compose.scale.yaml down -v")
        return None

    thread, reply, pending, _ = chat(base, f"Please refund order {order}. I confirm, go ahead.",
                                     customer)
    if not pending:  # some models ask "shall I go ahead?" first
        thread, reply, pending, _ = chat(base, "Yes, go ahead.", customer, thread)
    if not pending:
        print(f"the agent didn't request the refund: {reply[:80]!r}")
        return False
    print(f"refund for {order} is waiting for staff")

    # Two approvals at once, very likely on different copies.
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(
            lambda _: call(base, f"/api/staff/approvals/{thread}", {"approve": True}, staff=True),
            range(2)))
    for status, _, served_by in results:
        print(f"  approval on {served_by}: HTTP {status}")
    _, text, _ = call(base, "/api/staff/refunds", staff=True)
    times = sum(r["order_id"] == order for r in json.loads(text))
    print(f"{order} was paid {times} time(s)")
    return sorted(s for s, _, _ in results) == [200, 404] and times == 1


def check(base: str) -> None:
    base = base.rstrip("/")
    if call(base, "/healthz", timeout=5)[0] != 200:
        print(f"Nothing healthy at {base}. Is `docker compose -f compose.scale.yaml up` running?")
        sys.exit(1)
    results = {}
    for name, fn in [("requests spread across copies", check_spread),
                     ("conversation moves between copies", check_shared_conversation),
                     ("double approval pays once", check_double_approval)]:
        print(f"\n== {name}")
        started = time.perf_counter()
        results[name] = fn(base)
        print(f"   {'PASS' if results[name] else 'SKIP' if results[name] is None else 'FAIL'}"
              f"  ({time.perf_counter() - started:.1f}s)")
    sys.exit(0 if all(r is not False for r in results.values()) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a scaled-out Pedalwise deployment.")
    parser.add_argument("--check", metavar="URL", default="http://127.0.0.1:8080")
    check(parser.parse_args().check)
