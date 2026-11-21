"""
Tutorial 28 — Schema migrations: changing tables that already have data
========================================================================

Every table so far was made with CREATE TABLE IF NOT EXISTS at startup.
That works until the first change to an existing table. Tutorial 28 needs
one: disabling accounts means a new column, users.disabled_at. Existing
databases (your Docker volume, the shared Postgres) already have a users
table, so IF NOT EXISTS does nothing, the column never appears, and the
new code fails there with "no such column".

Migrations are the fix (pedalwise/migrations.py): numbered steps, applied
in order, recorded in the database itself, each in a transaction.

    1  baseline: the tables Tutorials 21-27 created
    2  users can be disabled            ALTER TABLE users ADD COLUMN ...
    3  index for the login throttle

What you'll learn:
  1. Why "create if missing" can't change a schema, and what replaces it
  2. Recording applied migrations, and catching edits with checksums
  3. All-or-nothing: each migration in a transaction
  4. Where to run them: at startup for one copy; as a separate step before
     the copies start when there are several (compose.scale.yaml's
     "migrate" service), with copies refusing an out-of-date schema
  5. Expand, then contract: changes old and new code can both live with
     during a deploy; forward-only, with backups instead of "down" steps

The first use of migration 2: 27_auth.py --disable-user, which blocks an
account's logins and ends all its sessions at once.

Run:
  uv run 28_migrations.py                 # upgrade an old database, step by step (no API calls)
  uv run 28_migrations.py --status        # the app's database: applied, pending, edited
  uv run 28_migrations.py --upgrade       # apply what's pending
In Docker:
  docker compose exec pedalwise python 28_migrations.py --status
"""

import argparse
import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path

from pedalwise import migrations
from pedalwise.config import Settings
from pedalwise.database import Database
from pedalwise.migrations import LATEST, status, upgrade


def print_status(state: dict) -> None:
    print(f"  schema version {state['version']} of {state['latest']}")
    for m in state["pending"]:
        print(f"  pending   {m.version}: {m.name}")
    for m in state["changed"]:
        print(f"  EDITED    {m.version}: {m.name}  (changed after it ran: add a new migration)")
    if state["unknown"]:
        print(f"  unknown   {state['unknown']}  (from newer code)")
    if not (state["pending"] or state["changed"] or state["unknown"]):
        print("  up to date")


async def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = await Database.connect(str(Path(tmp) / "old.sqlite"))

        print("1. A database as Tutorial 26's code left it: tables, data, no record")
        print("   of migrations, and no users table yet.")
        await db.execute("CREATE TABLE refunds (order_id TEXT PRIMARY KEY, customer TEXT NOT NULL,"
                         " amount_eur DOUBLE PRECISION NOT NULL, created TEXT NOT NULL)")
        await db.execute("INSERT INTO refunds VALUES ('A1001', 'sam@example.com', 899.0, "
                         "'2026-11-01 10:00:00')")
        print_status(await status(db))

        print("\n2. Upgrade. The baseline's IF NOT EXISTS skips the tables that are there;")
        print("   the rest are created, then the users table gets its new column.")
        await upgrade(db, log=lambda message: print(f"  {message}"))
        print_status(await status(db))
        rows = await db.fetchall("SELECT order_id, amount_eur FROM refunds")
        print(f"  the existing refund is still there: {rows}")

        print("\n3. Again: nothing to do. Every copy can run this at startup safely.")
        print(f"  applied this time: {await upgrade(db, log=print) or 'nothing'}")

        print("\n4. Someone edits migration 3 after it ran. The database won't get the")
        print("   edit, so status says so instead of silently drifting:")
        original = migrations.MIGRATIONS[2]
        migrations.MIGRATIONS[2] = replace(original, statements=(
            "CREATE INDEX IF NOT EXISTS login_failures_key ON login_failures(key)",))
        print_status(await status(db))
        migrations.MIGRATIONS[2] = original

        print("\n5. A migration that fails halfway changes nothing (a transaction):")
        broken = migrations.Migration(LATEST + 1, "broken on purpose", (
            "ALTER TABLE users ADD COLUMN nickname TEXT",
            "ALTER TABLE no_such_table ADD COLUMN oops TEXT"))
        migrations.MIGRATIONS.append(broken)
        try:
            await upgrade(db, log=lambda message: print(f"  {message}"))
        except Exception as error:
            print(f"  failed: {error}")
        finally:
            migrations.MIGRATIONS.remove(broken)
        columns = [r[0] for r in await db.fetchall("SELECT name FROM pragma_table_info('users')")]
        print(f"  users columns: {columns}")
        print(f"  'nickname' was rolled back: {'nickname' not in columns}")
        await db.close()


async def show_status() -> None:
    settings = Settings()
    db = await Database.connect(settings.db_url)
    print(f"Database: {'Postgres' if db.postgres else settings.db_url}")
    print_status(await status(db))
    await db.close()


async def run_upgrade() -> None:
    db = await Database.connect(Settings().db_url)
    ran = await upgrade(db)
    print(f"Applied {len(ran)} migration(s)." if ran else "Already up to date.")
    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema migrations for the Pedalwise app.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()
    asyncio.run(show_status() if args.status else run_upgrade() if args.upgrade else demo())
