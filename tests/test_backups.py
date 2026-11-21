"""Tests for Tutorial 29's backups: taken live, verified, restored, pruned.

On Postgres (TEST_DATABASE_URL) these need pg_dump and pg_restore of the
server's major version or newer; CI installs them.
"""

import asyncio
import os
import shutil

import pytest
from fastapi.testclient import TestClient

from pedalwise import backups as bk
from pedalwise.database import Database
from pedalwise.migrations import LATEST, status
from pedalwise.refunds import RefundLedger
from pedalwise.server import create_app

from .conftest import ScriptedModel, WordEmbeddings, login, make_settings

if os.getenv("TEST_DATABASE_URL") and not shutil.which("pg_dump"):
    pytest.skip("pg_dump isn't installed", allow_module_level=True)


def run(coro):
    return asyncio.run(coro)


def start_app(tmp_path, db_url, chat=False):
    """Start the real app once: tables and demo accounts, and with chat=True
    a saved conversation (LangGraph makes its tables on first use)."""
    app = create_app(make_settings(tmp_path, db_url), model=ScriptedModel(),
                     fallback_model=None, embeddings=WordEmbeddings())
    with TestClient(app) as client:
        if chat:
            login(client, "sam@example.com")
            assert client.post("/api/chat", json={"message": "my orders"}).status_code == 200


async def refunded(db_url) -> list[str]:
    db = await Database.connect(db_url)
    try:
        return [r["order_id"] for r in await RefundLedger(db).all()]
    finally:
        await db.close()


async def pay(db_url, order_id) -> bool:
    db = await Database.connect(db_url)
    try:
        return await RefundLedger(db).record(order_id, "sam@example.com", 10.0)
    finally:
        await db.close()


def drop_kept(db_url, kept):
    if bk.is_postgres_url(db_url):
        bk._drop_database(db_url, kept)


def test_a_backup_of_the_running_app_verifies(tmp_path, db_url):
    start_app(tmp_path, db_url, chat=True)
    run(pay(db_url, "A1001"))

    async def scenario():
        db = await Database.connect(db_url)  # "the app", still connected
        path = await bk.backup(db_url, tmp_path / "backups")
        report = await bk.verify(path, db_url)
        await db.close()
        return path, report
    path, report = run(scenario())
    assert report.ok, report.problems
    assert report.version == LATEST
    assert report.counts["refunds"] == 1 and report.counts["users"] == 3
    assert report.counts["checkpoints"] > 0  # the conversations are in it too
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert bk.backups(tmp_path / "backups") == [path]


def test_a_damaged_backup_is_refused(tmp_path, db_url):
    start_app(tmp_path, db_url)
    path = run(bk.backup(db_url, tmp_path / "backups"))
    with open(path, "r+b") as f:
        f.seek(f.seek(0, 2) // 2)
        f.write(b"bit rot")
    report = run(bk.verify(path, db_url))
    assert not report.ok and "checksum" in report.problems[0]
    with pytest.raises(ValueError):
        run(bk.restore(path, db_url, tmp_path / "backups", log=lambda m: None))


def test_restore_brings_back_the_backup_and_keeps_the_old_data(tmp_path, db_url):
    start_app(tmp_path, db_url)
    run(pay(db_url, "A1001"))
    path = run(bk.backup(db_url, tmp_path / "backups"))
    run(pay(db_url, "A1002"))  # after the backup: a restore loses it

    kept = run(bk.restore(path, db_url, tmp_path / "backups", log=lambda m: None))
    try:
        assert run(refunded(db_url)) == ["A1001"]
        assert run(pay(db_url, "A1001")) is False  # still known: can't be paid twice
        start_app(tmp_path, db_url)  # the app starts on the restored database
        # The data from before the restore wasn't destroyed.
        kept_url = (bk._with_database(db_url, kept) if bk.is_postgres_url(db_url)
                    else str(kept))
        assert run(refunded(kept_url)) == ["A1001", "A1002"]
    finally:
        drop_kept(db_url, kept)


def test_an_old_backup_is_migrated_on_restore(tmp_path, db_url):
    start_app(tmp_path, db_url)

    async def make_it_old():  # as if taken before migration 3 existed
        db = await Database.connect(db_url)
        await db.execute("DROP INDEX login_failures_key")
        await db.execute("DELETE FROM schema_migrations WHERE version = 3")
        await db.close()
    run(make_it_old())
    path = run(bk.backup(db_url, tmp_path / "backups"))
    report = run(bk.verify(path, db_url))
    assert report.ok and report.version == 2 and "restore() migrates it" in report.notes[0]

    kept = run(bk.restore(path, db_url, tmp_path / "backups", log=lambda m: None))
    try:
        async def version():
            db = await Database.connect(db_url)
            state = await status(db)
            await db.close()
            return state["version"]
        assert run(version()) == LATEST
    finally:
        drop_kept(db_url, kept)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="Postgres only")
def test_postgres_restore_refuses_while_the_app_is_connected(tmp_path, db_url):
    start_app(tmp_path, db_url)
    path = run(bk.backup(db_url, tmp_path / "backups"))

    async def scenario():
        db = await Database.connect(db_url)  # the app, still running
        try:
            with pytest.raises(RuntimeError, match="Stop the app"):
                await bk.restore(path, db_url, tmp_path / "backups", log=lambda m: None)
        finally:
            await db.close()
    run(scenario())
    assert run(refunded(db_url)) == []  # untouched, and no leftover databases:
    with bk._admin(db_url) as conn:
        names = [r[0] for r in conn.execute("SELECT datname FROM pg_database")]
    assert not [n for n in names if "_restoring_" in n or "_verify_" in n]


def test_sqlite_backup_needs_an_existing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(bk.backup(str(tmp_path / "missing.sqlite"), tmp_path / "backups"))
    assert not (tmp_path / "missing.sqlite").exists()  # and didn't create one
    assert bk.backups(tmp_path / "backups") == []


def test_prune_keeps_the_newest(tmp_path):
    folder = tmp_path / "backups"
    folder.mkdir()
    for day in range(1, 6):
        (folder / f"pedalwise-202611{day:02d}-030000.sqlite").write_text("x")
        (folder / f"pedalwise-202611{day:02d}-030000.sqlite.sha256").write_text("x")
    (folder / "before-restore-20261101-000000.sqlite").write_text("x")  # not a backup
    deleted = bk.prune(folder, keep=2)
    assert [p.name[10:18] for p in deleted] == ["20261101", "20261102", "20261103"]
    assert [p.name[10:18] for p in bk.backups(folder)] == ["20261104", "20261105"]
    assert (folder / "before-restore-20261101-000000.sqlite").exists()
    assert not (folder / "pedalwise-20261101-030000.sqlite.sha256").exists()
