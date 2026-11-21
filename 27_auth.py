"""
Tutorial 27 — Real logins
=========================

Since Tutorial 20 the app has said "a real app would use proper logins".
It identified customers by whatever email the browser sent, so anyone
could read Sam's orders by typing his address, and staff shared one token.
That's fine for a demo on your own machine and nowhere else.

This tutorial replaces it with accounts and sessions (pedalwise/auth.py):

  Who you are     comes from a login session, never from the request.
                  Every customer_email field is gone from the API.
  Passwords       stored as Argon2id hashes: salted and deliberately slow,
                  so a stolen table can't be cracked quickly.
  Sessions        a random id in a cookie that page scripts can't read
                  (HttpOnly), that other sites' forms don't send (SameSite),
                  and that only travels over HTTPS in production (Secure).
                  The database keeps only its hash, and logout deletes it.
  CSRF            every change also needs the session's CSRF token in a
                  header, which a page on another site can't read or send.
  Guessing        5 failed logins per email (20 per IP) in 15 minutes, then
                  a pause. Wrong email and wrong password get the same reply,
                  in the same time, so emails can't be probed.
  Staff           their own accounts. The audit log records which staff
                  member approved which refund. STAFF_TOKEN remains only
                  for scripts.

What you'll learn:
  1. Authentication (who are you?) versus authorization (may you do this?)
  2. Storing passwords and session ids so a database leak isn't a breach
  3. Cookie flags and CSRF tokens, and what each one stops
  4. Throttling and avoiding account enumeration
  5. An audit log for actions that move money

Setup:
  For the demo accounts (Sam, Dana, staff@example.com), set a password of
  12+ characters in .env; the app creates them at startup:
      PEDALWISE_DEMO_PASSWORD=...
  For real accounts, leave that unset and use --add-user.

Run:
  uv run 27_auth.py                       # try to break in (no API calls)
  uv run 27_auth.py --add-user you@example.com --name "You" --role staff
  uv run 27_auth.py --list                # accounts (never passwords)
  uv run 27_auth.py --audit               # who decided which refunds
  uv run 27_auth.py --set-password you@example.com    # and signs you out everywhere
  uv run 27_auth.py --disable-user dana@example.com   # and signs her out (Tutorial 28)

In Docker, run it inside the container, so it uses the container's database
(the volume, or Postgres in the scaled stack), not your local file:
  docker compose exec pedalwise python 27_auth.py --add-user you@example.com --role staff
  docker compose exec pedalwise python 27_auth.py --list
  docker compose -f compose.scale.yaml exec pedalwise python 27_auth.py --audit
"""

import argparse
import asyncio
import getpass
import logging
import secrets
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from pedalwise.auth import AuditLog, Users
from pedalwise.config import Settings
from pedalwise.database import Database
from pedalwise.migrations import upgrade
from pedalwise.server import SESSION_COOKIE, create_app


# ---------------------------------------------------------------------------
# The demo: the real app, a throwaway database, and an attacker's checklist
# ---------------------------------------------------------------------------
def show(label: str, response, expect: int) -> None:
    mark = "ok  " if response.status_code == expect else "FAIL"
    detail = response.json().get("detail", "") if response.headers.get(
        "content-type", "").startswith("application/json") else ""
    print(f"  {mark} {label:<52} HTTP {response.status_code} {detail}")


def demo() -> None:
    logging.getLogger("pedalwise").setLevel(logging.ERROR)  # no "Failed login" noise here
    password = secrets.token_urlsafe(12)  # demo accounts for this run only
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(db_path=str(Path(tmp) / "auth-demo.sqlite"), database_url="",
                            index_dir=Path(tmp) / "index", embeddings="none",
                            fallback_model="none", demo_password=password, staff_token="")
        with TestClient(create_app(settings)) as client:
            print("1. Nobody logged in")
            show("read my account", client.get("/api/me"), 401)
            show("chat", client.post("/api/chat", json={"message": "hi"}), 401)
            show("open the staff queue", client.get("/api/staff/approvals"), 401)

            print("\n2. Logging in as Sam")
            response = client.post("/api/login", json={"email": "sam@example.com",
                                                       "password": password})
            show("log in", response, 200)
            print(f"     cookie: {response.headers['set-cookie'].split(';', 1)[1].strip()}")
            csrf = response.json()["csrf_token"]

            print("\n3. Forged requests (another site making Sam's browser act)")
            show("change something without the CSRF token",
                 client.post("/api/feedback", json={"run_id": "00000000-0000-0000-0000-"
                                                    "000000000000", "score": 1}), 403)
            show("...with a made-up CSRF token",
                 client.post("/api/feedback", headers={"X-CSRF-Token": "guess"},
                             json={"run_id": "00000000-0000-0000-0000-000000000000",
                                   "score": 1}), 403)
            show("...with the real token (unknown run, so 404)",
                 client.post("/api/feedback", headers={"X-CSRF-Token": csrf},
                             json={"run_id": "00000000-0000-0000-0000-000000000000",
                                   "score": 1}), 404)

            print("\n4. Sam trying staff powers")
            show("open the staff queue", client.get("/api/staff/approvals"), 403)

            print("\n5. Probing for accounts")
            timings = {}
            for label, email in [("wrong password", "dana@example.com"),
                                 ("no such account", "nobody@example.com")]:
                times = []
                for _ in range(3):
                    started = time.perf_counter()
                    response = client.post("/api/login", json={"email": email,
                                                               "password": "wrong-password"})
                    times.append((time.perf_counter() - started) * 1000)
                timings[label] = statistics.median(times)
                show(f"{label} ({timings[label]:.0f} ms)", response, 401)
            print("     same message, similar time: an attacker can't tell which emails exist")

            print("\n6. Guessing Dana's password")
            for attempt in range(3):  # 3 more failures, 5 in all
                client.post("/api/login", json={"email": "dana@example.com", "password": "nope"})
            show("6th attempt, even with the right password",
                 client.post("/api/login", json={"email": "dana@example.com",
                                                 "password": password}), 429)

            print("\n7. Logging out")
            stolen = client.cookies[SESSION_COOKIE]
            client.post("/api/logout", headers={"X-CSRF-Token": csrf})
            client.cookies.set(SESSION_COOKIE, stolen)
            show("reuse the cookie someone copied before logout", client.get("/api/me"), 401)


# ---------------------------------------------------------------------------
# Managing accounts
# ---------------------------------------------------------------------------
async def add_user(email: str, name: str, role: str) -> None:
    # Typed, never passed as an argument: arguments end up in shell history.
    password = getpass.getpass("Password (12+ characters): ")
    if password != getpass.getpass("Again: "):
        raise SystemExit("The passwords don't match.")
    db = await Database.connect(Settings().db_url)
    users = Users(db)
    await upgrade(db, log=lambda message: None)  # tables up to date (28)
    try:
        user = await users.add(email, name, role, password)
        print(f"Created {user.role} account {user.email}.")
    except ValueError as error:
        raise SystemExit(str(error))
    except Exception as error:  # e.g. the email is taken (UNIQUE)
        raise SystemExit(f"Couldn't create the account: {error}")
    finally:
        await db.close()


async def set_password(email: str) -> None:
    """Change a password; the account's sessions end, everywhere."""
    password = getpass.getpass("New password (12+ characters): ")
    if password != getpass.getpass("Again: "):
        raise SystemExit("The passwords don't match.")
    db = await Database.connect(Settings().db_url)
    await upgrade(db, log=lambda message: None)
    try:
        if not await Users(db).set_password(email, password):
            raise SystemExit(f"No account {email}.")
    except ValueError as error:
        raise SystemExit(str(error))
    finally:
        await db.close()
    print(f"Changed the password for {email} and ended its sessions.")


async def set_disabled(email: str, disabled: bool) -> None:
    """Disabling also signs the account out everywhere (Tutorial 28)."""
    db = await Database.connect(Settings().db_url)
    await upgrade(db, log=lambda message: None)
    if not await Users(db).set_disabled(email, disabled):
        await db.close()
        raise SystemExit(f"No account {email}.")
    await AuditLog(db).record("cli", "account_disabled" if disabled else "account_enabled",
                              email=email.strip().lower())
    print(f"{'Disabled' if disabled else 'Enabled'} {email}"
          + (" and ended all its sessions." if disabled else "."))
    await db.close()


async def list_users() -> None:
    db = await Database.connect(Settings().db_url)
    await upgrade(db, log=lambda message: None)
    for email, name, role, created, disabled in await db.fetchall(
            "SELECT email, name, role, created, disabled_at FROM users ORDER BY role, email"):
        print(f"{role:<9} {email:<28} {name:<20} since {created}"
              + (f"  DISABLED {disabled}" if disabled else ""))
    await db.close()


async def show_audit() -> None:
    db = await Database.connect(Settings().db_url)
    audit = AuditLog(db)
    await upgrade(db, log=lambda message: None)
    entries = await audit.recent(50)
    for entry in entries:
        print(f"{entry['created']}  {entry['actor']:<22} {entry['action']:<16} {entry['detail']}")
    if not entries:
        print("Nothing yet: staff decisions on refunds appear here.")
    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logins for the Pedalwise app.")
    parser.add_argument("--add-user", metavar="EMAIL")
    parser.add_argument("--name", default="")
    parser.add_argument("--role", choices=["customer", "staff"], default="customer")
    parser.add_argument("--list", action="store_true", help="list accounts")
    parser.add_argument("--set-password", metavar="EMAIL", help="change it; ends sessions")
    parser.add_argument("--disable-user", metavar="EMAIL", help="block logins, end sessions")
    parser.add_argument("--enable-user", metavar="EMAIL")
    parser.add_argument("--audit", action="store_true", help="show the audit log")
    args = parser.parse_args()
    if args.add_user:
        asyncio.run(add_user(args.add_user, args.name or args.add_user.split("@")[0], args.role))
    elif args.set_password:
        asyncio.run(set_password(args.set_password))
    elif args.disable_user or args.enable_user:
        asyncio.run(set_disabled(args.disable_user or args.enable_user, bool(args.disable_user)))
    elif args.list:
        asyncio.run(list_users())
    elif args.audit:
        asyncio.run(show_audit())
    else:
        demo()
