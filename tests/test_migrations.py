"""Tests for Tutorial 28's migrations, and the first feature that needed one."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from pedalwise import migrations
from pedalwise.database import Database
from pedalwise.migrations import LATEST, status, upgrade
from pedalwise.server import create_app

from .conftest import ScriptedModel, WordEmbeddings, login, make_settings

QUIET = {"log": lambda message: None}


def run(coro):
    return asyncio.run(coro)


def test_a_fresh_database_gets_every_migration_once(db_url):
    async def scenario():
        db = await Database.connect(db_url)
        assert [m.version for m in await upgrade(db, **QUIET)] == list(range(1, LATEST + 1))
        assert await upgrade(db, **QUIET) == []  # the second time: nothing to do
        state = await status(db)
        assert state["version"] == LATEST and not state["pending"] and not state["changed"]
        await db.close()
    run(scenario())


def test_an_old_database_keeps_its_data(db_url):
    async def scenario():
        db = await Database.connect(db_url)
        # As Tutorial 27's code left it: users without disabled_at, no record.
        await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, "
                         "name TEXT NOT NULL, role TEXT NOT NULL, password_hash TEXT NOT NULL, "
                         "created TEXT NOT NULL)")
        await db.execute("INSERT INTO users VALUES ('1', 'sam@example.com', 'Sam', 'customer', "
                         "'hash', '2026-11-01 10:00:00')")
        await upgrade(db, **QUIET)
        assert await db.fetchall("SELECT email, disabled_at FROM users") == [
            ("sam@example.com", None)]
        await db.close()
    run(scenario())


def test_copies_racing_to_migrate_apply_each_step_once(db_url):
    async def scenario():
        dbs = [await Database.connect(db_url) for _ in range(3)]
        results = await asyncio.gather(*(upgrade(db, **QUIET) for db in dbs))
        assert sorted(m.version for ran in results for m in ran) == list(range(1, LATEST + 1))
        rows = await dbs[0].fetchall("SELECT version FROM schema_migrations ORDER BY version")
        assert [r[0] for r in rows] == list(range(1, LATEST + 1))
        for db in dbs:
            await db.close()
    run(scenario())


def test_a_failed_migration_changes_nothing(db_url, monkeypatch):
    broken = migrations.Migration(LATEST + 1, "broken", (
        "ALTER TABLE users ADD COLUMN nickname TEXT",
        "ALTER TABLE no_such_table ADD COLUMN oops TEXT"))

    async def scenario():
        db = await Database.connect(db_url)
        await upgrade(db, **QUIET)
        monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS + [broken])
        with pytest.raises(Exception):
            await upgrade(db, **QUIET)
        # The first statement was rolled back with the second.
        await db.execute("ALTER TABLE users ADD COLUMN nickname TEXT")
        assert (await status(db))["version"] == LATEST
        await db.close()
    run(scenario())


def test_edited_migrations_are_reported(db_url, monkeypatch):
    async def scenario():
        db = await Database.connect(db_url)
        await upgrade(db, **QUIET)
        edited = migrations.Migration(3, "index for the login throttle", ("SELECT 1",))
        monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:2] + [edited])
        assert [m.version for m in (await status(db))["changed"]] == [3]
        await db.close()
    run(scenario())


def test_the_app_refuses_an_out_of_date_schema(tmp_path, db_url):
    app = create_app(make_settings(tmp_path, db_url, migrate_on_start=False),
                     model=ScriptedModel(), fallback_model=None, embeddings=WordEmbeddings())
    with pytest.raises(RuntimeError, match="Run the migrations first"):
        with TestClient(app):
            pass


def test_disabling_an_account_signs_it_out_everywhere(client):
    login(client, "dana@example.com")
    assert client.get("/api/me").status_code == 200
    client.portal.call(client.app.state.users.set_disabled, "dana@example.com", True)
    assert client.get("/api/me").status_code == 401           # the open session is gone
    client.logged_in_as = None
    assert login(client, "dana@example.com").status_code == 401  # and she can't log back in
    client.portal.call(client.app.state.users.set_disabled, "dana@example.com", False)
    assert login(client, "dana@example.com").status_code == 200


def test_changing_a_password_signs_out_everywhere(client):
    login(client, "sam@example.com")
    client.portal.call(client.app.state.users.set_password, "sam@example.com",
                       "a-brand-new-password")
    assert client.get("/api/me").status_code == 401
    client.logged_in_as = None
    assert login(client, "sam@example.com").status_code == 401  # the old one is gone
    assert login(client, "sam@example.com", "a-brand-new-password").status_code == 200


def test_demo_accounts_follow_the_demo_password(tmp_path, db_url):
    def start(password):
        settings = make_settings(tmp_path, db_url)
        object.__setattr__(settings, "demo_password", password)
        return TestClient(create_app(settings, model=ScriptedModel(), fallback_model=None,
                                     embeddings=WordEmbeddings()))
    with start("the-first-password") as first:
        assert login(first, "sam@example.com", "the-first-password").status_code == 200
    with start("the-second-password") as second:  # PEDALWISE_DEMO_PASSWORD changed
        assert login(second, "sam@example.com", "the-second-password").status_code == 200
