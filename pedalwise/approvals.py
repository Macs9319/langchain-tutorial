"""Refunds waiting for a staff member, saved in the database (Tutorials 21, 26).

The paused agent itself lives in the checkpointer. This table is the list
of where to look. It used to be a dict in memory, which is fine until the
server restarts: then the list is gone, and every paused refund is stuck
forever. Deploys and container restarts happen all the time, so anything
that must outlive a request belongs in storage, not in a variable.

With several copies of the app (Tutorial 26) it also has to be *shared*:
the customer's request may reach one copy and the staff member's click
another.
"""

import json

from .database import Database, now_text


class ApprovalQueue:
    def __init__(self, db: Database):
        self._db = db

    async def setup(self) -> None:
        await self._db.execute("""CREATE TABLE IF NOT EXISTS pending_approvals (
            thread_id TEXT PRIMARY KEY,
            customer  TEXT NOT NULL,
            actions   TEXT NOT NULL,
            created   TEXT NOT NULL)""")

    async def add(self, thread_id: str, customer: str, actions: list[dict]) -> None:
        await self._db.execute(
            "INSERT INTO pending_approvals (thread_id, customer, actions, created) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (thread_id) DO UPDATE SET "
            "customer = excluded.customer, actions = excluded.actions",
            (thread_id, customer, json.dumps(actions), now_text()))

    async def take(self, thread_id: str) -> dict | None:
        """Remove and return one item. A single statement, so two staff
        members clicking at once, even on different copies of the app,
        can't both get it."""
        row = await self._db.fetchone(
            "DELETE FROM pending_approvals WHERE thread_id = ? RETURNING customer, actions",
            (thread_id,))
        return {"customer": row[0], "actions": json.loads(row[1])} if row else None

    async def all(self) -> list[dict]:
        rows = await self._db.fetchall(
            "SELECT thread_id, customer, actions FROM pending_approvals ORDER BY created")
        return [{"thread_id": t, "customer": c, "actions": json.loads(a)} for t, c, a in rows]

    async def has(self, thread_id: str) -> bool:
        return await self._db.fetchone(
            "SELECT 1 FROM pending_approvals WHERE thread_id = ?", (thread_id,)) is not None

    async def ping(self) -> None:
        """Raise if the database can't be read (used by /readyz)."""
        await self._db.fetchone("SELECT 1")
