"""Backups, checked by restoring them, and restores (Tutorial 29).

Everything the app knows lives in one database: accounts, the refund ledger,
the audit log, conversations, telemetry. Lose it and you lose more than
data: a refund ledger that forgets an order will happily pay it again.

    backup()   a consistent copy, plus a .sha256 file to detect damage later
    verify()   restore the copy somewhere disposable and check it: a backup
               you've never restored is a hope, not a backup
    restore()  put a backup back, without destroying what's there first
    prune()    keep the newest N

How each database is copied:
  SQLite    VACUUM INTO, which writes a consistent snapshot even while the
            app is writing. Copying the file is NOT safe: in WAL mode the
            newest changes sit in a separate -wal file, so a plain copy can
            be missing them, or be corrupt.
  Postgres  pg_dump, a consistent snapshot of a running database, in its
            compressed "custom" format, restored with pg_restore. pg_dump
            must be the server's major version or newer.

Backups hold everything the database holds, password hashes and customers'
questions included, so they get the same protection: files only their owner
can read, and an encrypted off-site copy in real life (a backup on the same
disk as the database dies with it).
"""

import asyncio
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .database import Database, is_postgres_url
from .migrations import status, upgrade

# Names look like pedalwise-20261121-031500.sqlite: they sort by time.
NAME = re.compile(r"^pedalwise-(\d{8}-\d{6})\.(sqlite|dump)$")
# The app's tables, and LangGraph's (the saved conversations).
TABLES = ["users", "refunds", "pending_approvals", "audit_log", "runs", "feedback",
          "sessions", "checkpoints"]


# ---------------------------------------------------------------------------
# Taking a backup
# ---------------------------------------------------------------------------
async def backup(db_url: str, backup_dir: Path | str) -> Path:
    """Write a consistent copy of the database into backup_dir."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = backup_dir / f"pedalwise-{stamp}.{'dump' if is_postgres_url(db_url) else 'sqlite'}"
    if path.exists():
        raise FileExistsError(f"{path.name} exists already (two backups in one second?)")
    partial = path.with_name(path.name + ".partial")  # never mistaken for a finished backup
    try:
        if is_postgres_url(db_url):
            await asyncio.to_thread(_pg, "pg_dump", db_url, "--format=custom", "--no-owner",
                                    f"--file={partial}")
        else:
            await asyncio.to_thread(_vacuum_into, db_url, partial)
        os.chmod(partial, 0o600)  # it holds password hashes and customers' data
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)
    _sidecar(path).write_text(f"{_sha256(path)}  {path.name}\n")
    os.chmod(_sidecar(path), 0o600)
    return path


def _vacuum_into(db_path: str, target: Path) -> None:
    if not Path(db_path).exists():  # sqlite3 would quietly create an empty one
        raise FileNotFoundError(f"No database at {db_path}")
    # Read-only: a backup must never be able to change the database.
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        source.execute("VACUUM INTO ?", (str(target),))
    finally:
        source.close()


# ---------------------------------------------------------------------------
# Checking a backup
# ---------------------------------------------------------------------------
@dataclass
class Report:
    path: Path
    version: int = 0                                  # schema version in the backup
    counts: dict[str, int | None] = field(default_factory=dict)  # None: no such table
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


async def verify(path: Path | str, db_url: str) -> Report:
    """Restore the backup into a throwaway database and check what came back.

    db_url is the live database: for Postgres, the scratch database is made
    on the same server (and dropped afterwards). The live one isn't touched.
    """
    path = Path(path)
    report = Report(path)
    if not _checksum_ok(path, report):
        return report
    if path.suffix == ".dump":
        scratch = _with_database(db_url, f"pedalwise_verify_{uuid.uuid4().hex[:8]}")
        try:
            await asyncio.to_thread(_create_database, db_url, _database_name(scratch))
            await asyncio.to_thread(_pg_restore, path, scratch)
            await _check(scratch, report)
        except subprocess.CalledProcessError as error:
            report.problems.append(f"pg_restore failed: {error.stderr.strip()[-300:]}")
        finally:
            await asyncio.to_thread(_drop_database, db_url, _database_name(scratch))
        return report
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "verify.sqlite"
        shutil.copyfile(path, copy)  # fine here: nothing has this file open
        try:
            with sqlite3.connect(copy) as conn:
                result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError as error:
            report.problems.append(f"not a readable SQLite database: {error}")
            return report
        if result != "ok":
            report.problems.append(f"integrity check: {result}")
            return report
        await _check(str(copy), report)
    return report


async def _check(url: str, report: Report) -> None:
    """What "this backup is usable" means for this app."""
    db = await Database.connect(url)
    try:
        for table in TABLES:
            try:
                report.counts[table] = (await db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0]
            except Exception:  # an old backup, from before that table existed
                report.counts[table] = None
        state = await status(db)
        report.version = state["version"]
        if report.version == 0:
            report.problems.append("no migrations recorded: is this the app's database?")
        if state["changed"]:
            report.problems.append(f"migrations edited since: {[m.version for m in state['changed']]}")
        if state["unknown"]:
            report.problems.append(f"made by newer code (migrations {state['unknown']}): "
                                   "restore it with that code")
        if state["pending"] and report.version:
            report.notes.append(f"schema {report.version} of {state['latest']}: "
                                "restore() migrates it")
        if report.counts.get("feedback") and report.counts.get("runs") is not None:
            orphans = (await db.fetchone(
                "SELECT COUNT(*) FROM feedback WHERE run_id NOT IN (SELECT run_id FROM runs)"))[0]
            if orphans:
                report.problems.append(f"{orphans} feedback rows point at missing runs")
        if report.counts.get("checkpoints") is None:
            report.notes.append("no saved conversations table (the app creates it on start)")
    finally:
        await db.close()


def _checksum_ok(path: Path, report: Report) -> bool:
    if not path.exists():
        report.problems.append("file not found")
        return False
    if not _sidecar(path).exists():
        report.problems.append(f"no {_sidecar(path).name}: can't tell if it's damaged")
        return False
    expected = _sidecar(path).read_text().split()[0]
    if _sha256(path) != expected:
        report.problems.append("checksum doesn't match: the file changed or is damaged")
        return False
    return True


# ---------------------------------------------------------------------------
# Restoring
# ---------------------------------------------------------------------------
async def restore(path: Path | str, db_url: str, backup_dir: Path | str, log=print) -> Path | str:
    """Replace the live database with a backup. Stop the app first.

    Nothing is destroyed: the backup is verified before anything changes,
    and the current database is kept (a file in backup_dir for SQLite, a
    renamed database for Postgres), in case the backup turns out wrong.
    Returns where the old data was kept. Then migrates the restored copy,
    since an older backup may be behind the code.
    """
    path = Path(path)
    report = await verify(path, db_url)
    if not report.ok:
        raise ValueError(f"Not restoring {path.name}: {'; '.join(report.problems)}")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if path.suffix == ".dump":
        live = _database_name(db_url)
        incoming = f"{live}_restoring_{stamp.replace('-', '_')}"
        kept = f"{live}_before_restore_{stamp.replace('-', '_')}"
        await asyncio.to_thread(_create_database, db_url, incoming)
        try:
            await asyncio.to_thread(_pg_restore, path, _with_database(db_url, incoming))
        except BaseException:
            await asyncio.to_thread(_drop_database, db_url, incoming)
            raise
        # The swap: two renames, which fail if anything is still connected.
        await asyncio.to_thread(_swap_databases, db_url, live, kept, incoming)
        log(f"Restored {path.name}. The previous database is kept as '{kept}'.")
    else:
        kept = Path(backup_dir) / f"before-restore-{stamp}.sqlite"
        if Path(db_url).exists():
            await asyncio.to_thread(_vacuum_into, db_url, kept)
            os.chmod(kept, 0o600)
        await asyncio.to_thread(_copy_into_sqlite, path, db_url)
        log(f"Restored {path.name}. " + (f"The previous database is kept as {kept}."
                                         if kept.exists() else "There was no database before."))
    db = await Database.connect(db_url)
    try:
        await upgrade(db, log=log)
    finally:
        await db.close()
    return kept


def _copy_into_sqlite(backup_file: Path, db_path: str) -> None:
    """SQLite's backup API, page by page into the live file, through SQLite
    itself. Copying over the file instead could leave an old -wal file next
    to it, which SQLite would then replay on top: corruption."""
    source = sqlite3.connect(f"file:{backup_file}?mode=ro", uri=True)
    target = sqlite3.connect(db_path, timeout=5)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()


# ---------------------------------------------------------------------------
# Keeping the newest N
# ---------------------------------------------------------------------------
def backups(backup_dir: Path | str) -> list[Path]:
    """Finished backups in backup_dir, oldest first."""
    folder = Path(backup_dir)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if NAME.match(p.name)),
                  key=lambda p: NAME.match(p.name).group(1))


def prune(backup_dir: Path | str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` backups. Returns what was deleted.

    Real setups often keep more of a spread (7 daily, 4 weekly, 12 monthly),
    because some problems are noticed weeks late. Newest-N keeps this short.
    """
    if keep < 1:
        raise ValueError("keep at least one backup")
    old = backups(backup_dir)[:-keep]
    for path in old:
        path.unlink()
        _sidecar(path).unlink(missing_ok=True)
    return old


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pg(tool: str, url: str, *args: str) -> None:
    """Run pg_dump or pg_restore. The connection goes in PG* environment
    variables, not the command line, where other users could see the
    password in the process list."""
    from psycopg.conninfo import conninfo_to_dict
    keys = {"host": "PGHOST", "port": "PGPORT", "user": "PGUSER", "password": "PGPASSWORD",
            "dbname": "PGDATABASE"}
    env = dict(os.environ)
    for key, value in conninfo_to_dict(url).items():
        if key in keys and value is not None:
            env[keys[key]] = str(value)
    if not shutil.which(tool):
        raise FileNotFoundError(f"{tool} isn't installed (it comes with the Postgres client)")
    subprocess.run([tool, *args], env=env, check=True, capture_output=True, text=True)


def _pg_restore(path: Path, url: str) -> None:
    # One transaction: a restore that fails halfway leaves nothing behind.
    _pg("pg_restore", url, "--no-owner", "--exit-on-error", "--single-transaction",
        f"--dbname={_database_name(url)}", str(path))


def _database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/")


def _with_database(url: str, name: str) -> str:
    return urlsplit(url)._replace(path=f"/{name}").geturl()


def _admin(url: str):
    """A connection to the server's maintenance database: you can't create,
    drop or rename the database you're connected to."""
    import psycopg
    return psycopg.connect(_with_database(url, "postgres"), autocommit=True)


def _quote(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"unexpected database name {name!r}")
    return f'"{name}"'


def _create_database(url: str, name: str) -> None:
    with _admin(url) as conn:
        conn.execute(f"CREATE DATABASE {_quote(name)}")


def _drop_database(url: str, name: str) -> None:
    with _admin(url) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {_quote(name)} WITH (FORCE)")


def _swap_databases(url: str, live: str, kept: str, incoming: str) -> None:
    import psycopg
    with _admin(url) as conn:
        try:
            conn.execute(f"ALTER DATABASE {_quote(live)} RENAME TO {_quote(kept)}")
        except psycopg.errors.ObjectInUse:
            conn.execute(f"DROP DATABASE {_quote(incoming)}")
            raise RuntimeError(f"Something is still connected to '{live}'. Stop the app "
                               "(and anything else using the database), then try again.")
        conn.execute(f"ALTER DATABASE {_quote(incoming)} RENAME TO {_quote(live)}")
