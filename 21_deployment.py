"""
Tutorial 21 — Deploying the capstone
====================================

Tutorial 20's app runs on your laptop. Deployed, it runs somewhere you can't
watch: in a container that gets stopped, restarted and replaced, with
settings you don't type in, logs you read later, and a platform that keeps
asking "are you alive?". This tutorial packages the Pedalwise app for that
and adds the small changes that make it safe to run there.

What changed in the app, and why:

    pedalwise/config.py     every setting comes from the environment, so the
                            same image runs anywhere; production refuses to
                            start with a missing key or a weak staff token
    pedalwise/approvals.py  refunds waiting for staff are saved in SQLite;
                            they used to be a dict and vanished on restart
    pedalwise/logs.py       one JSON object per log line, for log collectors
    pedalwise/server.py     /healthz and /readyz for the platform to poll,
                            no /docs in production, an audit log of refunds

And new files for running it:

    Dockerfile              builds a small image: locked dependencies with
                            uv, no dev tools, runs as a non-root user
    .dockerignore           keeps .env, databases and .venv out of the image
    compose.yaml            runs the image with a volume for the database
                            and the search index, restarting on crashes
    .github/workflows/ci.yml  runs the tests and builds the image on every push

What you'll learn:
  1. Twelve-factor config: settings from the environment, secrets never in
     the image, and failing fast when they're wrong
  2. Where state lives: anything that must survive a restart goes on a volume
  3. Health checks: liveness versus readiness, and why neither calls the model
  4. Structured logs and an audit trail
  5. Why this app runs as one process, and what you'd change to run many
  6. Smoke-testing a deployment from the outside

Why one process: conversations and the approval queue live in SQLite, a
file on one machine, and each process builds its own search index. One
uvicorn process handles many chats at once because everything is async
(Tutorial 11). To run several copies behind a load balancer, move the
conversations to Postgres (langgraph-checkpoint-postgres), the approval
queue to the same database, and the index to a shared vector store.

Setup:
  Same as Tutorial 20. For production, also set a long STAFF_TOKEN in .env:
    python -c "import secrets; print(secrets.token_urlsafe(32))"

Run:
  uv run 21_deployment.py --config         # show the settings, find problems
  uv run 21_deployment.py                  # serve, the way the container does
  PEDALWISE_ENV=production uv run 21_deployment.py   # JSON logs, strict checks

With Docker:
  docker compose up --build                # http://127.0.0.1:8001
  uv run 21_deployment.py --check http://127.0.0.1:8001
  uv run 21_deployment.py --check http://127.0.0.1:8001 --chat   # 1 model call
  docker compose down                      # the volume keeps the data
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

import uvicorn

from pedalwise.config import Settings
from pedalwise.logs import setup_logging
from pedalwise.server import create_app

logger = logging.getLogger("pedalwise")


# ---------------------------------------------------------------------------
# 1. Serve: what the container runs
# ---------------------------------------------------------------------------
def serve() -> None:
    settings = Settings()
    setup_logging(settings.log_level, json_lines=settings.production)

    # Fail fast: a server that starts with a missing key looks healthy and
    # then fails every chat. Crashing at startup makes the deploy fail
    # instead, where someone is watching.
    problems = settings.problems()
    for problem in problems:
        logger.error("Config problem: %s", problem)
    if problems and settings.production:
        sys.exit(1)

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        # Use the logging set up above instead of uvicorn's own.
        log_config=None,
        # Behind a proxy (nginx, a cloud load balancer) the client's address
        # arrives in X-Forwarded-For. Trust that header only from the proxy.
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        # On `docker stop` the container gets SIGTERM. Uvicorn stops taking
        # new requests and gives open chats this long to finish.
        timeout_graceful_shutdown=30,
    )


# ---------------------------------------------------------------------------
# 2. Show the settings, without the secrets
# ---------------------------------------------------------------------------
def show_config() -> None:
    settings = Settings()
    print("Settings (from environment variables and .env):\n")
    for name, value in vars(settings).items():
        if name in ("openai_api_key", "staff_token"):
            value = "(generated for this run)" if (
                name == "staff_token" and settings.staff_token_generated) else (
                "(set)" if value else "(not set)")
        if name != "staff_token_generated":
            print(f"  {name:<16} {value}")
    problems = settings.problems()
    print()
    if problems:
        print("Problems:" + "".join(f"\n  - {p}" for p in problems))
        sys.exit(1)
    print("No problems found.")


# ---------------------------------------------------------------------------
# 3. Smoke test a running deployment from the outside
# ---------------------------------------------------------------------------
def request(url: str, body: dict | None = None, timeout: float = 10) -> tuple[int, str]:
    """GET (or POST JSON) and return (status, text). Status 0: no connection."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except (urllib.error.URLError, TimeoutError) as error:
        return 0, str(error)


def check(base: str, chat: bool) -> None:
    base = base.rstrip("/")
    failures = 0

    def report(ok: bool | None, name: str, detail: str, started: float) -> None:
        nonlocal failures
        mark = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
        failures += ok is False
        print(f"  {mark}  {name:<22} {detail}  ({(time.perf_counter() - started) * 1000:.0f} ms)")

    print(f"Checking {base}\n")
    expectations = [
        # (name, path, expected status, why)
        ("alive", "/healthz", 200, "process answers"),
        ("ready", "/readyz", 200, "database reachable"),
        ("chat page", "/", 200, "page served"),
        ("staff API locked", "/api/staff/approvals", 401, "needs the staff token"),
    ]
    for name, path, expected, why in expectations:
        started = time.perf_counter()
        status, text = request(base + path)
        detail = why if status == expected else f"expected {expected}, got {status or text}"
        report(status == expected, name, detail, started)
        if name == "alive" and status == 0:
            print("\nNothing is listening there. Is the server running?")
            sys.exit(1)

    started = time.perf_counter()
    status, _ = request(base + "/docs")
    report(None if status == 200 else True, "API docs hidden",
           "hidden" if status != 200 else "visible: fine in development, not in production",
           started)

    if chat:
        # One real question through the whole stack: model, search, streaming.
        started = time.perf_counter()
        status, text = request(base + "/api/chat", {
            "message": "What are your opening hours?", "customer_email": "sam@example.com"},
            timeout=120)
        # Server-Sent Events: blocks separated by a blank line, each with
        # "event: ..." and "data: {json}" lines.
        events = []
        if status == 200:
            for block in text.strip().split("\n\n"):
                fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
                events.append((fields.get("event"), json.loads(fields.get("data", "{}"))))
        kinds = [kind for kind, _ in events]
        reply = "".join(data["text"] for kind, data in events if kind == "token")
        ok = bool(reply) and "error" not in kinds and kinds[-1:] == ["done"]
        report(ok, "chat answers", f"{reply[:60]!r}" if ok else f"status {status}, events {kinds}",
               started)

    print(f"\n{'All checks passed.' if not failures else f'{failures} check(s) failed.'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve, inspect or check the Pedalwise app.")
    parser.add_argument("--config", action="store_true", help="show settings and problems")
    parser.add_argument("--check", metavar="URL", help="smoke-test a running deployment")
    parser.add_argument("--chat", action="store_true",
                        help="with --check: also ask one real question (1 model call)")
    args = parser.parse_args()
    if args.config:
        show_config()
    elif args.check:
        check(args.check, args.chat)
    else:
        serve()
