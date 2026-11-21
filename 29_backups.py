"""
Tutorial 29 — Backups and restores
==================================

Tutorial 28 said "take a backup before migrating production" without saying
how. Since then the app's database holds things you can't get back:
accounts, the refund ledger, the audit log of who approved what, saved
conversations. It sits in one Docker volume, or one Postgres database, and
nothing else has a copy.

This tutorial adds pedalwise/backups.py:

  backup    a consistent copy while the app keeps running: SQLite's
            VACUUM INTO, or pg_dump for Postgres. Plus a checksum file.
  verify    restore it somewhere disposable and check it: the checksum, the
            database's own integrity check, the schema version, row counts.
  restore   verify first, keep the current database, swap in the backup,
            then run the migrations (an old backup can be behind the code).
  prune     keep the newest N.

What you'll learn:
  1. Why copying a live database file is not a backup (the demo shows it)
  2. A backup counts only once you've restored it
  3. What a restore loses: everything since the backup. For a refund
     ledger that means orders that can be refunded twice.
  4. Running backups on a schedule, next to the app (compose.yaml)
  5. Never destroying the only copy: keep the old database until the
     restored one is proven

What's not backed up: the search index (it's rebuilt from data/) and
secrets (.env), which belong in a secrets store with their own backups.

Run:
  uv run 29_backups.py                      # the demo, on a throwaway database (no API calls)
  uv run 29_backups.py --backup             # back up the app's database into backups/
  uv run 29_backups.py --list               # backups, newest last
  uv run 29_backups.py --verify NAME        # restore it to a scratch copy and check it
  uv run 29_backups.py --restore NAME       # stop the app first; asks to confirm
  uv run 29_backups.py --every 24h --keep 7 # back up forever, keeping the newest 7

In Docker the "backup" service runs that last one next to the app, into its
own volume (see compose.yaml). Restoring there:
  docker compose exec backup python 29_backups.py --list
  docker compose stop pedalwise
  docker compose exec backup python 29_backups.py --restore NAME
  docker compose start pedalwise
Scaled out (Postgres), the same with -f compose.scale.yaml; stop every copy.
"""

import argparse
import asyncio
import logging
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from pedalwise import backups as bk
from pedalwise.config import Settings
from pedalwise.database import Database
from pedalwise.migrations import upgrade
from pedalwise.refunds import RefundLedger

logger = logging.getLogger("pedalwise.backups")


def print_report(report: bk.Report) -> None:
    mark = "ok" if report.ok else "FAILED"
    print(f"  {report.path.name}: {mark}"
          + (f", schema version {report.version}" if report.counts else ""))
    if report.counts:
        print("  rows: " + ", ".join(f"{t} {n}" for t, n in report.counts.items()
                                     if n is not None))
    for problem in report.problems:
        print(f"  problem: {problem}")
    for note in report.notes:
        print(f"  note: {note}")


# ---------------------------------------------------------------------------
# The demo: a throwaway database, a disaster, and getting it back
# ---------------------------------------------------------------------------
async def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        live = str(Path(tmp) / "pedalwise.sqlite")
        folder = Path(tmp) / "backups"

        print("1. The app is running: WAL mode, like the real one (LangGraph turns")
        print("   it on), and a refund paid for order A1001.")
        db = await Database.connect(live)
        await db.execute("PRAGMA journal_mode=WAL")
        await upgrade(db, log=lambda message: None)
        ledger = RefundLedger(db)
        await ledger.record("A1001", "sam@example.com", 899.0)
        wal = Path(live + "-wal")
        print(f"   pedalwise.sqlite     {Path(live).stat().st_size:>8,} bytes")
        print(f"   pedalwise.sqlite-wal {wal.stat().st_size:>8,} bytes  <- the newest changes")

        print("\n2. The obvious backup: copy the file while the app runs.")
        copy = Path(tmp) / "copied.sqlite"
        shutil.copyfile(live, copy)
        try:
            with sqlite3.connect(copy) as conn:
                found = conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
            print(f"   the copy has {found} refunds")
        except sqlite3.Error as error:
            print(f"   the copy is useless: {error}")
        print("   Everything still in the -wal file is missing. (Copying both files")
        print("   isn't safe either: they can change between the two copies.)")

        print("\n3. A real backup: VACUUM INTO writes a consistent snapshot, live.")
        first = await bk.backup(live, folder)
        print(f"   wrote {first.name} and {first.name}.sha256")

        print("\n4. Is it any good? Restore it to a scratch copy and look:")
        print_report(await bk.verify(first, live))

        print("\n5. Life goes on: A1002 is refunded after the backup.")
        await ledger.record("A1002", "dana@example.com", 120.0)
        print(f"   ledger: {[r['order_id'] for r in await ledger.all()]}")

        print("\n6. Disaster: a bad script empties the refunds table.")
        await db.execute("DELETE FROM refunds")
        print(f"   ledger: {[r['order_id'] for r in await ledger.all()]}")
        print("   The ledger now thinks nothing was ever refunded, so every order")
        print("   can be refunded again. Restore. Step one: stop the app.")
        await db.close()
        kept = await bk.restore(first, live, folder, log=lambda message: print(f"   {message}"))

        print("\n7. After the restore (the app started again):")
        db = await Database.connect(live)
        ledger = RefundLedger(db)
        print(f"   ledger: {[r['order_id'] for r in await ledger.all()]}")
        print(f"   pay A1001 again? refused: {not await ledger.record('A1001', 'sam', 899.0)}")
        print(f"   pay A1002 again? allowed: {await ledger.record('A1002', 'dana', 120.0)}"
              "  <- paid twice")
        await db.close()
        print("   A restore brings back the moment of the backup, and loses everything")
        print("   after it: here, a refund that really happened. So after restoring,")
        print("   compare the ledger with the payment provider's records (the real source")
        print("   for money), and back up often enough that the gap is small. Postgres")
        print("   can also archive its write-ahead log continuously, to restore to any")
        print("   second rather than to last night (point-in-time recovery).")
        print(f"   The pre-restore database is kept too: {Path(kept).name}")

        print("\n8. A backup damaged on disk is caught before it's used:")
        damaged = folder / "pedalwise-20260101-000000.sqlite"
        shutil.copyfile(first, damaged)
        shutil.copyfile(bk._sidecar(first), bk._sidecar(damaged))
        with open(damaged, "r+b") as f:
            f.seek(f.seek(0, 2) // 2)  # a few bytes in the middle, as a failing disk might
            f.write(b"bit rot")
        print_report(await bk.verify(damaged, live))

        print("\n9. Keeping the newest N (a nightly job with --keep 7):")
        for day in range(2, 11):  # nine more, pretending a night apart
            path = folder / f"pedalwise-202611{day:02d}-030000.sqlite"
            shutil.copyfile(first, path)
            shutil.copyfile(bk._sidecar(first), bk._sidecar(path))
        deleted = bk.prune(folder, keep=7)
        print(f"   deleted {len(deleted)}, kept {[p.name[10:18] for p in bk.backups(folder)]}")


# ---------------------------------------------------------------------------
# Working on the app's real database
# ---------------------------------------------------------------------------
def find(settings: Settings, name: str) -> Path:
    """A backup by file name (looked up in the backup folder) or path."""
    path = Path(name)
    if not path.exists():
        path = settings.backup_dir / name
    if not path.exists():
        raise SystemExit(f"No backup {name}. See --list.")
    return path


async def take_backup(settings: Settings, keep: int | None) -> bool:
    started = time.perf_counter()
    path = await bk.backup(settings.db_url, settings.backup_dir)
    report = await bk.verify(path, settings.db_url)
    print_report(report)
    if not report.ok:
        # Set aside, so prune and --list never count it as a good backup.
        for p in (path, bk._sidecar(path)):
            p.rename(p.with_name(p.name + ".failed"))
        logger.error("Backup failed verification: %s", "; ".join(report.problems))
        return False
    print(f"  {path.stat().st_size:,} bytes in {time.perf_counter() - started:.1f} s")
    if keep:
        for old in bk.prune(settings.backup_dir, keep):
            print(f"  deleted {old.name} (keeping the newest {keep})")
    return True


async def schedule(settings: Settings, every: float, keep: int) -> None:
    """Back up, then sleep. A loop is enough for a container next to the
    app; elsewhere use cron, a systemd timer or a Kubernetes CronJob. Any of
    them needs an alert when backups stop: a job that has silently failed
    for a month is worse than none, because everyone thinks it works."""
    print(f"Backing up every {every / 3600:g} h into {settings.backup_dir}, keeping {keep}.")
    while True:
        try:
            await take_backup(settings, keep)
        except Exception as error:  # keep trying: the database may just be restarting
            logger.error("Backup failed: %s", error)
            print(f"  backup failed: {error}", file=sys.stderr)
        sys.stdout.flush()
        await asyncio.sleep(every)


async def run_restore(settings: Settings, name: str, yes: bool) -> None:
    path = find(settings, name)
    where = "the Postgres database" if settings.database_url else settings.db_url
    print(f"This replaces {where} with {path.name}.")
    print("Everything written since that backup is lost from the live database")
    print("(the current one is kept, so nothing is deleted). Stop the app first.")
    if not yes and input("Type RESTORE to go on: ").strip() != "RESTORE":
        raise SystemExit("Not restored.")
    try:
        await bk.restore(path, settings.db_url, settings.backup_dir)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
    print("Done. Start the app, then check refunds against the payment provider.")


def list_backups(settings: Settings) -> None:
    found = bk.backups(settings.backup_dir)
    for path in found:
        print(f"  {path.name}  {path.stat().st_size:>10,} bytes")
    print(f"{len(found)} backup(s) in {settings.backup_dir}" if found
          else f"No backups in {settings.backup_dir} yet: try --backup.")


def duration(text: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if not text or text[-1] not in units:
        raise argparse.ArgumentTypeError("like 30m, 24h or 1d")
    return float(text[:-1]) * units[text[-1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backups for the Pedalwise app.")
    parser.add_argument("--backup", action="store_true", help="back up now, and check it")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", metavar="NAME")
    parser.add_argument("--restore", metavar="NAME")
    parser.add_argument("--yes", action="store_true", help="restore without asking")
    parser.add_argument("--every", type=duration, metavar="24h", help="back up on a schedule")
    parser.add_argument("--keep", type=int, metavar="N", help="delete all but the newest N")
    args = parser.parse_args()
    settings = Settings()
    if args.every:
        asyncio.run(schedule(settings, args.every, args.keep or 7))
    elif args.backup:
        sys.exit(0 if asyncio.run(take_backup(settings, args.keep)) else 1)
    elif args.list:
        list_backups(settings)
    elif args.verify:
        report = asyncio.run(bk.verify(find(settings, args.verify), settings.db_url))
        print_report(report)
        sys.exit(0 if report.ok else 1)
    elif args.restore:
        asyncio.run(run_restore(settings, args.restore, args.yes))
    else:
        asyncio.run(demo())
