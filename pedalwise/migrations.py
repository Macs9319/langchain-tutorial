"""Versioned schema migrations for the app's own tables (Tutorial 28).

Until Tutorial 27 each store ran CREATE TABLE IF NOT EXISTS at startup. That
creates missing tables but can never *change* one: add a column to the code
and every existing database (your Docker volume, the shared Postgres) still
lacks it, so the new code fails there.

Migrations fix that. Each one is a numbered, named list of statements. The
database records which numbers it has (table schema_migrations), and
upgrade() applies the missing ones in order, each in a transaction, under
the lock from Tutorial 26 so two copies can't run them at once.

Rules that keep this safe:
  - Never edit a migration that has run anywhere: add a new one. A checksum
    of each applied migration is stored, and status() reports edits.
  - Forward only. To undo something, write another migration. Take a
    backup before migrating production (29_backups.py --backup).
  - Expand, then contract. During a deploy, old and new copies of the app
    run side by side against the same database. So add first (a column the
    old code ignores), ship the code that uses it, and only remove things
    in a later migration, once no running code needs them.

LangGraph's checkpoint tables are not here: its setup() has its own
migrations (table checkpoint_migrations).
"""

import hashlib
from dataclasses import dataclass

from .database import Database, now_text


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        return hashlib.sha256("\n;\n".join(self.statements).encode()).hexdigest()[:16]


MIGRATIONS = [
    # 1: the tables Tutorials 21-27 created with CREATE TABLE IF NOT EXISTS.
    # Still "IF NOT EXISTS", so it also applies cleanly to databases made
    # before migrations existed: they already have these tables.
    Migration(1, "baseline: tables from tutorials 21-27", (
        """CREATE TABLE IF NOT EXISTS pending_approvals (
            thread_id TEXT PRIMARY KEY,
            customer  TEXT NOT NULL,
            actions   TEXT NOT NULL,
            created   TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS refunds (
            order_id   TEXT PRIMARY KEY,
            customer   TEXT NOT NULL,
            amount_eur DOUBLE PRECISION NOT NULL,
            created    TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS runs (
            run_id        TEXT PRIMARY KEY,
            thread_id     TEXT NOT NULL,
            customer      TEXT NOT NULL,
            kind          TEXT NOT NULL,
            outcome       TEXT NOT NULL,
            latency_ms    INTEGER NOT NULL,
            model_calls   INTEGER NOT NULL,
            input_tokens  INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            tools         TEXT NOT NULL,
            tool_errors   INTEGER NOT NULL,
            question      TEXT,
            answer        TEXT,
            created       TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS feedback (
            run_id  TEXT PRIMARY KEY REFERENCES runs(run_id),
            score   INTEGER NOT NULL,
            comment TEXT,
            created TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS runs_created ON runs(created)",
        """CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            email         TEXT NOT NULL UNIQUE,
            name          TEXT NOT NULL,
            role          TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created       TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            csrf_token TEXT NOT NULL,
            expires    TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS login_failures (
            key     TEXT NOT NULL,
            created TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS audit_log (
            actor   TEXT NOT NULL,
            action  TEXT NOT NULL,
            detail  TEXT NOT NULL,
            created TEXT NOT NULL)""",
    )),
    # 2: a real change to an existing table. Nullable, so code from before
    # this migration keeps working while copies are being replaced.
    Migration(2, "users can be disabled", (
        "ALTER TABLE users ADD COLUMN disabled_at TEXT",
    )),
    # 3: the login throttle counts failures per key and time on every login.
    Migration(3, "index for the login throttle", (
        "CREATE INDEX IF NOT EXISTS login_failures_key ON login_failures(key, created)",
    )),
]
LATEST = MIGRATIONS[-1].version


async def _ensure_table(db: Database) -> None:
    await db.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
        version  INTEGER PRIMARY KEY,
        name     TEXT NOT NULL,
        checksum TEXT NOT NULL,
        applied  TEXT NOT NULL)""")


async def applied(db: Database) -> dict[int, tuple[str, str, str]]:
    """{version: (name, checksum, applied at)} for what this database has."""
    await _ensure_table(db)
    rows = await db.fetchall("SELECT version, name, checksum, applied FROM schema_migrations")
    return {version: (name, checksum, at) for version, name, checksum, at in rows}


async def status(db: Database) -> dict:
    """Where this database stands compared with this code."""
    done = await applied(db)
    known = {m.version: m for m in MIGRATIONS}
    return {
        "version": max(done, default=0),
        "latest": LATEST,
        "pending": [m for m in MIGRATIONS if m.version not in done],
        # Edited after running somewhere: the database didn't get the edit.
        "changed": [m for m in MIGRATIONS if m.version in done
                    and done[m.version][1] != m.checksum],
        # The database is newer than this code (a rollback, or old copies
        # still running during a deploy): fine if those were additive.
        "unknown": sorted(v for v in done if v not in known),
    }


async def upgrade(db: Database, log=print) -> list[Migration]:
    """Apply every pending migration, in order. Safe to run from several
    copies at once, and safe to run again: the second time it does nothing."""
    ran = []
    async with db.schema_lock():
        # Read inside the lock: another copy may have just finished.
        done = await applied(db)
        for migration in MIGRATIONS:
            if migration.version in done:
                continue
            async with db.transaction() as tx:
                # Check again inside the transaction. On Postgres the lock
                # already rules out a race; on SQLite the lock does nothing,
                # but the transaction holds the write lock, so this is the
                # check that counts (another process may have just done it).
                if await tx.fetchone("SELECT 1 FROM schema_migrations WHERE version = ?",
                                     (migration.version,)):
                    continue
                log(f"Applying migration {migration.version}: {migration.name}")
                for statement in migration.statements:
                    await tx.execute(statement)
                await tx.execute(
                    "INSERT INTO schema_migrations (version, name, checksum, applied) "
                    "VALUES (?, ?, ?, ?)",
                    (migration.version, migration.name, migration.checksum, now_text()))
            ran.append(migration)
    return ran
