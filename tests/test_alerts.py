"""Tests for Tutorial 30's alerts: facts, rules, and messages once per change."""

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from pedalwise.alerts import RULES, Alerter, Facts, Finding, evaluate, gather, webhook

from .conftest import login

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


def run(coro):
    return asyncio.run(coro)


def metrics(runs=0, errors=0, p95=1000, tokens=0, up=0, down=0) -> dict:
    """The shape Telemetry.metrics() returns, with just what the rules read."""
    return {"runs": runs, "outcomes": {"ok": runs - errors, "error": errors},
            "latency_ms": {"p50": p95, "p95": p95 if runs else None},
            "tokens": {"input": tokens, "output": 0},
            "feedback": {"up": up, "down": down}}


def healthy(**changes) -> Facts:
    facts = Facts(now=NOW, app_url="http://app", metrics=metrics(runs=50),
                  backup_dir=Path("/backups"), newest_backup=NOW - timedelta(hours=3))
    for key, value in changes.items():
        setattr(facts, key, value)
    return facts


def by_rule(findings: list[Finding]) -> dict[str, Finding]:
    return {f.rule: f for f in findings}


def test_a_healthy_app_fires_nothing():
    findings = evaluate(healthy())
    assert [f.rule for f in findings] == [r.name for r in RULES]
    assert not [f for f in findings if f.firing is not False]


def test_rules_fire_past_their_limits():
    found = by_rule(evaluate(healthy(
        app_error="no answer", metrics=metrics(runs=50, errors=10, p95=45_000, tokens=2_000_000,
                                               up=2, down=8),
        approvals_waiting=1, oldest_approval=NOW - timedelta(hours=5),
        newest_backup=NOW - timedelta(days=2))))
    assert [name for name, f in found.items() if f.firing] == [
        "app_up", "error_rate", "slow_replies", "token_spend", "thumbs_down",
        "approvals_waiting", "backups"]
    assert "10 of 50 runs failed" in found["error_rate"].summary


def test_rates_need_enough_data():
    # 1 failure in 2 runs is 50%, but not an outage.
    found = by_rule(evaluate(healthy(metrics=metrics(runs=2, errors=1, p95=90_000, down=1))))
    assert not found["error_rate"].firing and "too few" in found["error_rate"].summary
    assert not found["slow_replies"].firing and not found["thumbs_down"].firing


def test_a_failed_backup_fires_even_if_an_older_one_is_fresh():
    facts = healthy(newest_failed_backup=NOW - timedelta(hours=1))
    assert by_rule(evaluate(facts))["backups"].firing
    facts.newest_failed_backup = NOW - timedelta(hours=5)  # an older failure, since fixed
    assert not by_rule(evaluate(facts))["backups"].firing


def test_unreadable_database_makes_data_rules_unknown_not_ok():
    found = by_rule(evaluate(healthy(database_error="can't connect", metrics=None)))
    assert found["database"].firing is True
    assert found["error_rate"].firing is None and found["approvals_waiting"].firing is None
    assert found["backups"].firing is False  # doesn't need the database


def test_unconfigured_checks_are_left_out():
    found = by_rule(evaluate(healthy(app_url=None, backup_dir=None)))
    assert "app_up" not in found and "backups" not in found


def test_a_broken_rule_fires():
    facts = healthy(metrics={"runs": 50})  # missing keys: the data rules raise
    found = by_rule(evaluate(facts))
    assert found["token_spend"].firing and "check itself failed" in found["token_spend"].summary


def finding(rule="error_rate", firing=True) -> Finding:
    return Finding(rule, "urgent", firing, "summary")


def test_one_message_per_change():
    alerter = Alerter(confirm=2, repeat_every=timedelta(hours=4))
    minute = timedelta(minutes=1)
    states = []
    for i, firing in enumerate([True, True, True, True, False, False]):
        states += [a.state for a in run(alerter.process([finding(firing=firing)], NOW + i * minute))]
    assert states == ["firing", "resolved"]  # confirmed on the second check, then one each way


def test_a_single_blip_sends_nothing():
    alerter = Alerter(confirm=2)
    assert run(alerter.process([finding()], NOW)) == []
    assert run(alerter.process([finding(firing=False)], NOW)) == []
    assert run(alerter.process([finding()], NOW)) == []  # the count started again


def test_reminders_while_it_lasts():
    alerter = Alerter(confirm=1, repeat_every=timedelta(hours=4))
    sent = [run(alerter.process([finding()], NOW + timedelta(hours=h))) for h in (0, 1, 4, 5, 8)]
    assert [[a.state for a in batch] for batch in sent] == [
        ["firing"], [], ["still firing"], [], ["still firing"]]
    assert "for 8 h 0 min" in sent[-1][0].text and "RUNBOOK.md#error_rate" in sent[-1][0].text


def test_unknown_neither_fires_nor_resolves():
    alerter = Alerter(confirm=1)
    run(alerter.process([finding()], NOW))
    assert run(alerter.process([finding(firing=None)], NOW)) == []
    assert "error_rate" in alerter.firing


def test_undelivered_alerts_are_retried_in_order():
    answers = iter([503, 200, 200])
    received = []

    def service(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.read())["text"])
        return httpx.Response(next(answers))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(service)) as client:
            alerter = Alerter(send=webhook("https://hook.example", client), confirm=1)
            first = await alerter.process([finding("app_up"), finding("database")], NOW)
            assert first == [] and [a.rule for a in alerter.outbox] == ["app_up", "database"]
            second = await alerter.flush()
            assert [a.rule for a in second] == ["app_up", "database"] and not alerter.outbox

    run(scenario())
    assert len(received) == 3 and received[0].startswith("[FIRING] app_up, urgent")


def test_gather_reads_the_running_app(client, tmp_path, db_url):
    login(client, "sam@example.com")
    assert client.post("/api/chat", json={"message": "Refund order A1001 please"}).status_code == 200
    backups = tmp_path / "backups"
    backups.mkdir()

    def ready(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if request.url.path == "/readyz" else 404)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(ready)) as http:
            return await gather(db_url, backups, "http://app", http)

    facts = run(scenario())
    assert facts.app_error is None and facts.database_error is None
    assert facts.metrics["runs"] == 1 and facts.approvals_waiting == 1
    found = by_rule(evaluate(facts))
    assert found["backups"].firing  # none taken yet
    (backups / f"pedalwise-{datetime.now(UTC):%Y%m%d-%H%M%S}.sqlite").touch()
    assert not by_rule(evaluate(run(gather(db_url, backups))))["backups"].firing


def test_gather_without_a_database(tmp_path):
    facts = run(gather(str(tmp_path / "missing.sqlite")))
    assert "no database file" in facts.database_error
    assert not (tmp_path / "missing.sqlite").exists()  # not created as a side effect


def test_every_rule_has_a_runbook_section():
    runbook = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    sections = set(re.findall(r"^## (\w+)$", runbook, re.MULTILINE))
    assert {r.name for r in RULES} <= sections

