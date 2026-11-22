"""Alerts: hearing about a problem before a customer tells you (Tutorial 30).

Tutorial 22 records what the app does and 29 backs it up, but both wait for
someone to look. Nobody looks at 3 a.m., or on the day the backup job
quietly starts failing. This module looks for you, every minute, and sends
a message when something needs a person.

It works in three steps, kept apart so each can be tested on its own:

    gather()    collect facts: is /readyz answering, the last hour's runs
                (error rate, p95, tokens, thumbs-down), refunds waiting
                for staff, the newest backup
    evaluate()  apply the rules to those facts: pure functions, no I/O
    Alerter     turn findings into messages: one when a problem starts, a
                reminder while it lasts, one when it's over. Never one per
                check, or people learn to ignore the channel.

Rules worth having are few. Each one here answers "would someone need to
act on this?", says what limit it compared against, and has a section in
RUNBOOK.md saying what to do. "urgent" means act now; "warning" means
during working hours. If an alert fires and nobody needs to do anything,
change the rule or delete it.

Three traps, each with a rule or a design choice against it:
  Silence isn't health. A dead app records no failed runs, so its error
    rate looks fine. The app_up rule asks /readyz from outside instead.
  A check that can't run isn't a passing check. If the database can't be
    read, the rules that need it are "unknown": they neither fire nor
    resolve, and the database rule fires instead.
  Who watches the watcher? If this process dies, or the webhook stops
    delivering, no alert says so. Ping a heartbeat URL after each good
    round (ALERT_HEARTBEAT_URL, e.g. healthchecks.io): that service alerts
    you when the pings stop.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from . import backups as bk
from .database import Database, is_postgres_url
from .telemetry import Telemetry

logger = logging.getLogger("pedalwise.alerts")

URGENT, WARNING = "urgent", "warning"
RUNBOOK = "RUNBOOK.md"


@dataclass(frozen=True)
class Limits:
    """Where each rule draws the line. Start from your normal numbers (see
    22_observability.py --report) with room to spare, then adjust: an alert
    that fires every week for nothing gets ignored, and then so does the
    one that matters."""
    error_rate: float = 0.10       # share of runs in the last hour that failed
    min_runs: int = 10             # fewer runs than this: too few to judge a rate
    p95_ms: int = 30_000           # 1 reply in 20 slower than this
    tokens_per_hour: int = 1_000_000  # a runaway loop, or someone abusing the chat
    thumbs_down: float = 0.5       # share of rated answers
    min_ratings: int = 5
    approval_wait: timedelta = timedelta(hours=4)  # a customer waiting on staff
    backup_age: timedelta = timedelta(hours=26)    # a daily backup, with slack


@dataclass
class Facts:
    """What gather() found. None means "couldn't find out" (or not configured)."""
    now: datetime
    app_url: str | None = None
    app_error: str | None = None        # set when /readyz didn't answer 200
    database_error: str | None = None   # set when the database couldn't be read
    metrics: dict | None = None         # the last hour, from Telemetry.metrics (22)
    approvals_waiting: int = 0
    oldest_approval: datetime | None = None
    backup_dir: Path | None = None
    newest_backup: datetime | None = None
    newest_failed_backup: datetime | None = None


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    firing: bool | None   # None: unknown, because its facts couldn't be collected
    summary: str


@dataclass(frozen=True)
class Alert:
    """A message to send."""
    rule: str
    severity: str
    state: str            # "firing", "still firing" or "resolved"
    summary: str
    since: datetime
    at: datetime

    @property
    def text(self) -> str:
        lasted = "" if self.state == "firing" else f" (for {_ago(self.at - self.since)})"
        return (f"[{self.state.upper()}] {self.rule}, {self.severity}{lasted}: {self.summary}\n"
                f"What to do: {RUNBOOK}#{self.rule}")


def _ago(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


# ---------------------------------------------------------------------------
# 1. Gathering facts (all the I/O is here)
# ---------------------------------------------------------------------------
async def gather(db_url: str, backup_dir: Path | str | None = None, app_url: str | None = None,
                 client: httpx.AsyncClient | None = None) -> Facts:
    facts = Facts(now=datetime.now(UTC), app_url=app_url,
                  backup_dir=Path(backup_dir) if backup_dir else None)
    if app_url:
        async with _client(client) as http:
            facts.app_error = await _probe(http, app_url)
    try:
        db = await _connect(db_url)
    except Exception as error:
        facts.database_error = f"can't connect: {error}"
    else:
        try:
            facts.metrics = await Telemetry(db).metrics(hours=1)
            oldest, waiting = await db.fetchone(
                "SELECT MIN(created), COUNT(*) FROM pending_approvals")
            facts.approvals_waiting = waiting
            facts.oldest_approval = _parse(oldest, "%Y-%m-%d %H:%M:%S")
        except Exception as error:
            facts.database_error = f"can't read: {error!r}"
            facts.metrics = None
        finally:
            await db.close()
    if facts.backup_dir:
        good = bk.backups(facts.backup_dir)
        facts.newest_backup = _backup_time(good[-1].name) if good else None
        # 29_backups.py renames a backup that fails verification to *.failed.
        failed = sorted(filter(None, (_backup_time(p.name) for p in
                                      facts.backup_dir.glob("pedalwise-*.failed"))))
        facts.newest_failed_backup = failed[-1] if failed else None
    return facts


def _client(client: httpx.AsyncClient | None):
    """The caller's client (tests pass one), or a new one closed afterwards."""
    if client is not None:
        return nullcontext(client)
    return httpx.AsyncClient()


async def _probe(http: httpx.AsyncClient, app_url: str) -> str | None:
    """Ask /readyz, as the platform's health check does (Tutorial 21), but
    from outside: this also catches a broken network or load balancer."""
    url = app_url.rstrip("/") + "/readyz"
    try:
        response = await http.get(url, timeout=5)
    except httpx.HTTPError as error:
        return f"no answer from {url} ({type(error).__name__})"
    return None if response.status_code == 200 else f"{url} answered {response.status_code}"


async def _connect(db_url: str) -> Database:
    # sqlite3 creates a missing file, which would hide a wrong path or a lost
    # volume behind an empty, perfectly healthy-looking database.
    if not is_postgres_url(db_url) and not Path(db_url).exists():
        raise FileNotFoundError(f"no database file at {db_url}")
    return await Database.connect(db_url)


def _parse(text: str | None, fmt: str) -> datetime | None:
    return datetime.strptime(text, fmt).replace(tzinfo=UTC) if text else None


def _backup_time(name: str) -> datetime | None:
    found = re.match(r"pedalwise-(\d{8}-\d{6})\.", name)
    return _parse(found.group(1), "%Y%m%d-%H%M%S") if found else None


# ---------------------------------------------------------------------------
# 2. The rules: facts in, (firing, summary) out
# ---------------------------------------------------------------------------
def app_up(f: Facts, limits: Limits) -> tuple[bool, str]:
    return (True, f.app_error) if f.app_error else (False, f"{f.app_url} is ready")


def database(f: Facts, limits: Limits) -> tuple[bool, str]:
    return (True, f.database_error) if f.database_error else (False, "readable")


def error_rate(f: Facts, limits: Limits) -> tuple[bool, str]:
    m = f.metrics
    failed = m["outcomes"].get("error", 0) + m["outcomes"].get("timeout", 0)
    if m["runs"] < limits.min_runs:
        # 1 failure in 2 runs is 50%, and means nothing. (And 0 runs could
        # mean the app is down: that's app_up's job.)
        return False, f"{failed} of {m['runs']} runs failed in the last hour (too few to judge)"
    rate = failed / m["runs"]
    return rate > limits.error_rate, (f"{failed} of {m['runs']} runs failed in the last hour "
                                      f"({rate:.0%}; the limit is {limits.error_rate:.0%})")


def slow_replies(f: Facts, limits: Limits) -> tuple[bool, str]:
    m = f.metrics
    p95 = m["latency_ms"]["p95"]
    if m["runs"] < limits.min_runs or p95 is None:
        return False, f"{m['runs']} runs in the last hour (too few to judge)"
    return p95 > limits.p95_ms, (f"p95 reply time {p95 / 1000:.1f} s over the last hour "
                                 f"(the limit is {limits.p95_ms / 1000:.0f} s)")


def token_spend(f: Facts, limits: Limits) -> tuple[bool, str]:
    used = f.metrics["tokens"]["input"] + f.metrics["tokens"]["output"]
    return used > limits.tokens_per_hour, (f"{used:,} tokens in the last hour "
                                           f"(the limit is {limits.tokens_per_hour:,})")


def thumbs_down(f: Facts, limits: Limits) -> tuple[bool, str]:
    votes = f.metrics["feedback"]
    rated = votes["up"] + votes["down"]
    if rated < limits.min_ratings:
        return False, f"{rated} answers rated in the last hour (too few to judge)"
    share = votes["down"] / rated
    return share > limits.thumbs_down, (f"{votes['down']} of {rated} rated answers got a "
                                        f"thumbs-down ({share:.0%}; the limit is "
                                        f"{limits.thumbs_down:.0%})")


def approvals_waiting(f: Facts, limits: Limits) -> tuple[bool, str]:
    if not f.oldest_approval:
        return False, "no refunds waiting for staff"
    waited = f.now - f.oldest_approval
    return waited > limits.approval_wait, (
        f"{f.approvals_waiting} refund(s) waiting for staff, the oldest for {_ago(waited)} "
        f"(the limit is {_ago(limits.approval_wait)})")


def backups(f: Facts, limits: Limits) -> tuple[bool, str]:
    if f.newest_backup is None:
        return True, f"no good backup in {f.backup_dir}"
    age = f.now - f.newest_backup
    if f.newest_failed_backup and f.newest_failed_backup > f.newest_backup:
        return True, (f"the latest backup failed verification; the newest good one is "
                      f"{_ago(age)} old")
    return age > limits.backup_age, (f"the newest good backup is {_ago(age)} old "
                                     f"(the limit is {_ago(limits.backup_age)})")


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    # Which facts it reads: the probe ("app"), the database's contents
    # ("data"), the backup folder ("backups"), or none of those ("").
    needs: str
    test: Callable[[Facts, Limits], tuple[bool, str]]


RULES = [
    Rule("app_up", URGENT, "app", app_up),
    Rule("database", URGENT, "", database),
    Rule("error_rate", URGENT, "data", error_rate),
    Rule("slow_replies", WARNING, "data", slow_replies),
    Rule("token_spend", WARNING, "data", token_spend),
    Rule("thumbs_down", WARNING, "data", thumbs_down),
    Rule("approvals_waiting", WARNING, "data", approvals_waiting),
    # A warning, not urgent: nothing is lost yet, but fix it today.
    Rule("backups", WARNING, "backups", backups),
]


def evaluate(facts: Facts, limits: Limits = Limits()) -> list[Finding]:
    found = []
    for rule in RULES:
        if rule.needs == "app" and not facts.app_url:
            continue  # not configured
        if rule.needs == "backups" and not facts.backup_dir:
            continue
        if rule.needs == "data" and facts.database_error:
            found.append(Finding(rule.name, rule.severity, None,
                                 "not checked: the database couldn't be read"))
            continue
        try:
            firing, summary = rule.test(facts, limits)
        except Exception as error:  # a broken rule must not look like a passing one
            firing, summary = True, f"the check itself failed: {error!r}"
        found.append(Finding(rule.name, rule.severity, firing, summary))
    return found


# ---------------------------------------------------------------------------
# 3. From findings to messages
# ---------------------------------------------------------------------------
Send = Callable[[Alert], Awaitable[None]]


@dataclass
class Alerter:
    """Remembers what's firing, so people get one message per change.

    confirm: how many checks in a row must fail before the first message.
    One failed probe is often a blip (a deploy, a restart); two a minute
    apart is a problem. Higher means fewer false alarms but later news.

    State is kept in memory: restart the watcher and it re-announces what's
    still firing. Twice is better than never.
    """
    send: Send | None = None
    confirm: int = 2
    repeat_every: timedelta = timedelta(hours=4)
    firing: dict[str, datetime] = field(default_factory=dict)    # rule -> since
    failing: dict[str, int] = field(default_factory=dict)        # rule -> checks in a row
    reminded: dict[str, datetime] = field(default_factory=dict)  # rule -> last message
    outbox: list[Alert] = field(default_factory=list)            # not delivered yet

    async def process(self, findings: list[Finding], now: datetime | None = None) -> list[Alert]:
        """Update the state and send what changed. Returns what was delivered."""
        now = now or datetime.now(UTC)
        for f in findings:
            if f.firing is None:  # unknown: no change either way
                continue
            if f.firing:
                self.failing[f.rule] = self.failing.get(f.rule, 0) + 1
                if f.rule not in self.firing and self.failing[f.rule] >= self.confirm:
                    self.firing[f.rule] = self.reminded[f.rule] = now
                    self._queue(Alert(f.rule, f.severity, "firing", f.summary, now, now))
                elif f.rule in self.firing and now - self.reminded[f.rule] >= self.repeat_every:
                    self.reminded[f.rule] = now
                    self._queue(Alert(f.rule, f.severity, "still firing", f.summary,
                                      self.firing[f.rule], now))
            else:
                self.failing.pop(f.rule, None)
                if f.rule in self.firing:
                    since = self.firing.pop(f.rule)
                    self.reminded.pop(f.rule, None)
                    self._queue(Alert(f.rule, f.severity, "resolved", f.summary, since, now))
        return await self.flush()

    def _queue(self, alert: Alert) -> None:
        # Logged once, when it happens: with JSON logs (21) a log platform can
        # alert on these lines too, and they're the record of the incident.
        level = logging.INFO if alert.state == "resolved" else logging.ERROR
        logger.log(level, "Alert %s: %s", alert.state, alert.rule,
                   extra={"alert": alert.rule, "state": alert.state,
                          "severity": alert.severity, "summary": alert.summary})
        self.outbox.append(alert)

    async def flush(self) -> list[Alert]:
        """Send what's waiting, in order. If sending fails, the rest wait for
        the next round: an alert lost because Slack was down for a minute is
        exactly the one you needed."""
        sent = []
        while self.outbox:
            if self.send is not None:
                try:
                    await self.send(self.outbox[0])
                except Exception as error:
                    logger.error("Couldn't deliver alert %s: %r (will retry)",
                                 self.outbox[0].rule, error)
                    break
            sent.append(self.outbox.pop(0))
        return sent


def webhook(url: str, client: httpx.AsyncClient | None = None) -> Send:
    """Post each alert as {"text": ...}: Slack's incoming webhooks, Mattermost
    and many chat tools accept that shape. For a pager service (PagerDuty,
    Opsgenie) or Discord ({"content": ...}), change the payload here."""
    async def send(alert: Alert) -> None:
        async with _client(client) as http:
            response = await http.post(url, json={"text": alert.text}, timeout=10)
            response.raise_for_status()
    return send


async def heartbeat(url: str, client: httpx.AsyncClient | None = None) -> bool:
    """Tell an outside service "the watcher ran, and could deliver". It
    alerts you when these stop arriving: the one failure this process can't
    report itself."""
    try:
        async with _client(client) as http:
            (await http.get(url, timeout=10)).raise_for_status()
        return True
    except httpx.HTTPError as error:
        logger.error("Heartbeat failed: %r", error)
        return False
