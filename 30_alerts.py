"""
Tutorial 30 — Alerts: hearing about problems before customers do
================================================================

Tutorial 22 records what the app does, and 29 backs it up every night. Both
wait for someone to look. Nobody does at 3 a.m., or in the week the backup
job quietly starts failing, so the first you hear of a problem is from an
angry customer, or when you need a backup that isn't there.

This tutorial adds pedalwise/alerts.py, which looks for you:

  gather     facts, every minute: does /readyz answer, the last hour's
             error rate, p95 and tokens (22), thumbs-down share, refunds
             waiting for staff, the age of the newest good backup (29)
  evaluate   rules, each with a limit, a severity (urgent: act now;
             warning: working hours) and a section in RUNBOOK.md
  Alerter    one message when a problem starts, reminders while it lasts,
             one when it ends, posted to a webhook (Slack and friends)

What you'll learn:
  1. Alerting on what needs a person, not on everything you can measure
  2. One message per change, confirmed by two checks, never one per check
  3. Rates need enough data: 1 failure in 2 runs isn't a 50% outage
  4. Silence isn't health: a dead app has no errors. Probe it from outside.
  5. A check that can't run is "unknown", not "ok"
  6. Who watches the watcher: retrying delivery, and a heartbeat that an
     outside service expects every minute
  7. A runbook: what each alert means and what to do, written before 3 a.m.

Run:
  uv run 30_alerts.py                       # the demo: an incident, start to end (no API calls)
  uv run 30_alerts.py --check               # check the app's database and backups once
  uv run 30_alerts.py --every 1m            # keep checking and send alerts
  uv run 30_alerts.py --test-notification   # prove the webhook works before you need it

Settings (in .env): ALERT_WEBHOOK_URL, where alerts are posted;
ALERT_HEARTBEAT_URL, pinged after every round; ALERT_APP_URL, the app's
address for /readyz. Without a webhook, alerts only go to the log.

In Docker the "alerts" service runs --every 1m next to the app (compose.yaml
and compose.scale.yaml):
  docker compose logs -f alerts
  docker compose exec alerts python 30_alerts.py --check
  docker compose exec alerts python 30_alerts.py --test-notification

After an incident, write down what happened, how you found out and how
long it took, and fix the cause with a test (20), an eval case (10, 25) or
a better rule here. An incident nobody learns from comes back.
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from pedalwise import backups as bk
from pedalwise.alerts import Alert, Alerter, Finding, evaluate, gather, heartbeat, webhook
from pedalwise.config import Settings
from pedalwise.database import Database, now_text
from pedalwise.logs import setup_logging
from pedalwise.migrations import upgrade
from pedalwise.telemetry import RunRecorder, Telemetry

logger = logging.getLogger("pedalwise.alerts")


def print_findings(findings: list[Finding], only: set[str] | None = None) -> None:
    for f in findings:
        if only is None or f.rule in only:
            mark = {True: "FIRING", False: "ok", None: "unknown"}[f.firing]
            print(f"     {mark:<8} {f.rule:<18} {f.summary}")


async def show(alert: Alert) -> None:
    """The demo's "webhook": print the message a person would get."""
    print("     >>> sent: " + alert.text.replace("\n", "\n               "))


# ---------------------------------------------------------------------------
# The demo: a throwaway database, an outage, and the alerts it causes
# ---------------------------------------------------------------------------
async def add_runs(telemetry: Telemetry, count: int, outcome: str = "ok",
                   latency_ms: int = 4000, tokens: int = 1500) -> None:
    """Pretend chat turns, recorded the way the server records them (22)."""
    for _ in range(count):
        recorder = RunRecorder()
        recorder.started = time.perf_counter() - latency_ms / 1000
        recorder.model_calls, recorder.input_tokens, recorder.output_tokens = 2, tokens, tokens // 10
        await telemetry.record(str(uuid.uuid4()), "demo", "sam@example.com", "chat", outcome,
                               recorder, "Where is my order?", "On its way.")


async def an_hour_passes(db: Database, clock: list[datetime]) -> None:
    """Move every recorded run two hours back, out of the rules' window."""
    await db.execute("UPDATE runs SET created = ?", (now_text(timedelta(hours=2)),))
    clock[0] += timedelta(hours=1)


async def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        live = str(Path(tmp) / "pedalwise.sqlite")
        folder = Path(tmp) / "backups"
        db = await Database.connect(live)
        await upgrade(db, log=lambda message: None)
        telemetry = Telemetry(db)
        alerter = Alerter(send=show)  # confirm=2: two failed checks before a message
        # The demo prints what happens; the real watcher logs it (JSON in production).
        logging.getLogger("pedalwise.alerts").disabled = True
        clock = [datetime.now(UTC)]  # the alerter's time: one minute per check

        async def check(note: str, only: set[str] | None = None, **where) -> list[Finding]:
            clock[0] += timedelta(minutes=1)
            print(f"\n   check: {note}")
            findings = evaluate(await gather(where.get("db_url", live),
                                             where.get("backup_dir", folder),
                                             where.get("app_url")))
            print_findings(findings, only)
            sent = await alerter.process(findings, now=clock[0])
            if not sent and any(f.firing for f in findings):
                print("     (nothing sent)")
            return findings

        print("1. A normal hour: 30 chats, nightly backup taken. Every rule is quiet.")
        await add_runs(telemetry, 30)
        await bk.backup(live, folder)
        await check("all fine")

        print("\n2. The model provider starts failing: 12 chats end in errors.")
        await add_runs(telemetry, 12, outcome="error", latency_ms=60_000)
        await check("first bad check", {"error_rate", "slow_replies"})
        print("     One bad check could be a blip (a deploy, a restart). Wait for a second.")
        await check("a minute later", {"error_rate"})
        await check("and another minute later", {"error_rate"})
        print("     Still failing, still one message. An alert per check would mean 60")
        print("     messages an hour, and people mute the channel.")

        print("\n3. Fixed: 30 good chats. But the rate covers the whole last hour...")
        await add_runs(telemetry, 30)
        await check("right after the fix", {"error_rate"})
        print("     ...so it keeps firing until the failures age out. Longer windows")
        print("     mean fewer false alarms, and slower to fire and to resolve.")
        await an_hour_passes(db, clock)
        await add_runs(telemetry, 20)
        await check("an hour later", {"error_rate", "slow_replies"})

        print("\n4. Silence isn't health. The app crashes, so nothing is recorded at all:")
        await an_hour_passes(db, clock)
        await check("no runs", {"error_rate"})
        print("     No runs, no failures: the error rate can't see a dead app. Ask it")
        print("     directly, from outside, the way a customer reaches it:")
        down = "http://127.0.0.1:9"  # nothing listens on port 9
        for note in ("app down", "app still down"):
            await check(note, {"app_up"}, app_url=down)

        print("\n5. A check that can't run isn't a passing check. The volume is gone:")
        gone = str(Path(tmp) / "missing.sqlite")
        await check("no database", {"database", "error_rate", "token_spend"}, db_url=gone)
        await check("still no database", {"database"}, db_url=gone)
        print("     The rules that need the data are unknown: they don't fire, and they")
        print("     don't count as fixed either.")

        print("\n6. Backups (29): the newest good one must be under 26 hours old.")
        newest = bk.backups(folder)[-1]
        stale = folder / f"pedalwise-{(datetime.now(UTC) - timedelta(days=2)):%Y%m%d-%H%M%S}.sqlite"
        newest.rename(stale)  # pretend the job stopped two days ago
        print_findings(evaluate(await gather(live, folder)), {"backups"})
        (folder / f"pedalwise-{datetime.now(UTC):%Y%m%d-%H%M%S}.sqlite.failed").touch()
        print("   and once last night's backup failed its verification:")
        print_findings(evaluate(await gather(live, folder)), {"backups"})

        print("\n7. Who watches the watcher? The webhook is down when an alert fires:")
        answers = iter([503, 200])
        received = []

        def chat_service(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return httpx.Response(next(answers))

        client = httpx.AsyncClient(transport=httpx.MockTransport(chat_service))
        paging = Alerter(send=webhook("https://chat.example/hook", client), confirm=1)
        findings = evaluate(await gather(live, folder))
        await paging.process(findings)
        print(f"   delivery failed (503); waiting to retry: {[a.rule for a in paging.outbox]}")
        await paging.flush()
        print(f"   next round: delivered, outbox {paging.outbox}. The webhook got:")
        print(f"     {received[-1]}")
        await client.aclose()
        print("   Retrying covers a short outage. It can't cover this process dying, or")
        print("   the machine it runs on: no alert can report its own absence. So after")
        print("   every round it pings ALERT_HEARTBEAT_URL, and a service such as")
        print("   healthchecks.io alerts you when the pings stop.")
        print("\n   Each alert links to its section of RUNBOOK.md: what it means, what to")
        print("   look at first, and how to fix it. Write those before you need them.")
        await db.close()


# ---------------------------------------------------------------------------
# Watching the real app
# ---------------------------------------------------------------------------
async def check_once(settings: Settings) -> bool:
    facts = await gather(settings.db_url, settings.backup_dir, settings.alert_app_url or None)
    findings = evaluate(facts)
    print_findings(findings)
    return all(f.firing is False for f in findings)


async def watch(settings: Settings, every: float) -> None:
    send = webhook(settings.alert_webhook_url) if settings.alert_webhook_url else None
    print(f"Checking every {every:g} s. Alerts go to "
          + ("the webhook and the log." if send else "the log only (no ALERT_WEBHOOK_URL)."))
    alerter = Alerter(send=send)
    while True:
        try:
            facts = await gather(settings.db_url, settings.backup_dir,
                                 settings.alert_app_url or None)
            await alerter.process(evaluate(facts))
            # Only when everything was delivered: a heartbeat says "I'm
            # running and I can reach you", not "all is well".
            if settings.alert_heartbeat_url and not alerter.outbox:
                await heartbeat(settings.alert_heartbeat_url)
        except Exception:
            logger.exception("Alert check failed")
        sys.stdout.flush()
        await asyncio.sleep(every)


async def test_notification(settings: Settings) -> bool:
    ok = True
    if not settings.alert_webhook_url:
        print("ALERT_WEBHOOK_URL isn't set.")
        ok = False
    else:
        now = datetime.now(UTC)
        alert = Alert("test", "warning", "firing",
                      "a test from 30_alerts.py --test-notification. If you can read this, "
                      "alerts reach you.", now, now)
        try:
            await webhook(settings.alert_webhook_url)(alert)
            print("Sent a test alert. Check that it arrived.")
        except httpx.HTTPError as error:
            print(f"Sending failed: {error!r}")
            ok = False
    if settings.alert_heartbeat_url:
        sent = await heartbeat(settings.alert_heartbeat_url)
        print("Heartbeat sent." if sent else "Heartbeat failed.")
        ok = ok and sent
    return ok


def duration(text: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600}
    if not text or text[-1] not in units:
        raise argparse.ArgumentTypeError("like 30s, 1m or 1h")
    return float(text[:-1]) * units[text[-1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alerts for the Pedalwise app.")
    parser.add_argument("--check", action="store_true", help="check once; exit 1 if not all ok")
    parser.add_argument("--every", type=duration, metavar="1m", help="keep checking, send alerts")
    parser.add_argument("--test-notification", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    setup_logging(settings.log_level, json_lines=settings.production)
    if args.every:
        asyncio.run(watch(settings, args.every))
    elif args.check:
        sys.exit(0 if asyncio.run(check_once(settings)) else 1)
    elif args.test_notification:
        sys.exit(0 if asyncio.run(test_notification(settings)) else 1)
    else:
        asyncio.run(demo())
