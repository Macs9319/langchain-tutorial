"""Accounts, sessions, login throttling and an audit log (Tutorial 27).

Until now the app believed whatever email the browser sent, so anyone could
read Sam's orders by typing his address, and staff shared one token. This
module is what replaces that:

  Users       email, name, role ("customer" or "staff") and a password
              hash. Argon2id: slow on purpose, salted, so a stolen table
              can't be reversed quickly. Never the password itself.
  Sessions    after login, a random session id in a cookie. The database
              stores only its SHA-256, so a leaked table can't be replayed
              as live sessions. Each session also has a CSRF token.
  Throttle    failed logins counted per email and per IP, in the database,
              so every copy of the app sees the same count (Tutorial 26).
  AuditLog    who did what: refunds decided, by which staff member.

All of it lives in the shared database (pedalwise/database.py).
"""

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from .database import Database, now_text

SESSION_DAYS = 7
MAX_FAILURES = 5                        # failed logins per email ...
MAX_FAILURES_PER_IP = 20                # ... and per IP address ...
FAILURE_WINDOW = timedelta(minutes=15)  # ... within this long

_hasher = PasswordHasher()  # argon2id with the library's current recommended settings
# Checked against when the email doesn't exist, so a wrong email takes as
# long as a wrong password. Otherwise timing would reveal which emails
# have accounts.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str
    role: str  # "customer" or "staff"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Users:
    def __init__(self, db: Database):
        self._db = db

    async def setup(self) -> None:
        await self._db.execute("""CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            email         TEXT NOT NULL UNIQUE,
            name          TEXT NOT NULL,
            role          TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created       TEXT NOT NULL)""")

    async def add(self, email: str, name: str, role: str, password: str) -> User:
        if role not in ("customer", "staff"):
            raise ValueError("role must be customer or staff")
        if len(password) < 12:
            raise ValueError("passwords need at least 12 characters")
        user = User(secrets.token_hex(8), email.strip().lower(), name, role)
        await self._db.execute(
            "INSERT INTO users (id, email, name, role, password_hash, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user.id, user.email, name, role, _hasher.hash(password), now_text()))
        return user

    async def get(self, email: str) -> User | None:
        row = await self._db.fetchone("SELECT id, email, name, role FROM users WHERE email = ?",
                                      (email.strip().lower(),))
        return User(*row) if row else None

    async def check_password(self, email: str, password: str) -> User | None:
        """The user if the password is right; None for a wrong email or password alike."""
        row = await self._db.fetchone(
            "SELECT id, email, name, role, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),))
        try:
            _hasher.verify(row[4] if row else _DUMMY_HASH, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return None
        if row is None:
            return None
        if _hasher.check_needs_rehash(row[4]):  # settings got stronger since this was saved
            await self._db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                                   (_hasher.hash(password), row[0]))
        return User(*row[:4])


class Sessions:
    def __init__(self, db: Database):
        self._db = db

    async def setup(self) -> None:
        await self._db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            csrf_token TEXT NOT NULL,
            expires    TEXT NOT NULL)""")

    async def create(self, user: User) -> tuple[str, str]:
        """A new session: (token for the cookie, CSRF token for the page)."""
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires = now_text(timedelta(days=-SESSION_DAYS))  # "now minus -7 days": a week ahead
        await self._db.execute(
            "INSERT INTO sessions (token_hash, user_id, csrf_token, expires) VALUES (?, ?, ?, ?)",
            (_digest(token), user.id, csrf, expires))
        return token, csrf

    async def get(self, token: str) -> tuple[User, str] | None:
        """(user, csrf token) for a live session, or None."""
        row = await self._db.fetchone(
            "SELECT u.id, u.email, u.name, u.role, s.csrf_token FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires > ?",
            (_digest(token), now_text()))
        return (User(*row[:4]), row[4]) if row else None

    async def delete(self, token: str) -> None:
        await self._db.execute("DELETE FROM sessions WHERE token_hash = ?", (_digest(token),))

    async def delete_expired(self) -> None:
        await self._db.execute("DELETE FROM sessions WHERE expires <= ?", (now_text(),))


class LoginThrottle:
    """Slows down password guessing.

    Per email: 5 failures in 15 minutes locks that account's logins for a
    while. Per IP: a higher limit, because many people can share one address
    (an office, a phone network), and one guesser shouldn't lock them all out.
    """

    def __init__(self, db: Database):
        self._db = db

    async def setup(self) -> None:
        await self._db.execute("""CREATE TABLE IF NOT EXISTS login_failures (
            key     TEXT NOT NULL,
            created TEXT NOT NULL)""")

    async def _count(self, key: str) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) FROM login_failures WHERE key = ? AND created > ?",
            (key, now_text(FAILURE_WINDOW)))
        return row[0]

    async def blocked(self, email: str, ip: str) -> bool:
        return (await self._count(f"email:{email}") >= MAX_FAILURES
                or await self._count(f"ip:{ip}") >= MAX_FAILURES_PER_IP)

    async def failed(self, email: str, ip: str) -> None:
        for key in (f"email:{email}", f"ip:{ip}"):
            await self._db.execute("INSERT INTO login_failures (key, created) VALUES (?, ?)",
                                   (key, now_text()))

    async def succeeded(self, email: str) -> None:
        await self._db.execute("DELETE FROM login_failures WHERE key = ?", (f"email:{email}",))


class AuditLog:
    """An append-only record of who did what. Logs rotate away; this doesn't."""

    def __init__(self, db: Database):
        self._db = db

    async def setup(self) -> None:
        await self._db.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            actor   TEXT NOT NULL,
            action  TEXT NOT NULL,
            detail  TEXT NOT NULL,
            created TEXT NOT NULL)""")

    async def record(self, actor: str, action: str, **detail) -> None:
        await self._db.execute(
            "INSERT INTO audit_log (actor, action, detail, created) VALUES (?, ?, ?, ?)",
            (actor, action, json.dumps(detail), now_text()))

    async def recent(self, limit: int = 50) -> list[dict]:
        rows = await self._db.fetchall(
            "SELECT actor, action, detail, created FROM audit_log ORDER BY created DESC LIMIT ?",
            (limit,))
        return [{"actor": a, "action": b, "detail": json.loads(d), "created": c}
                for a, b, d, c in rows]
