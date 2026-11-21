"""Refunds waiting for a staff member, saved in SQLite (Tutorial 21).

The paused agent itself lives in the checkpointer. This table is the list
of where to look. It used to be a dict in memory, which is fine until the
server restarts: then the list is gone, and every paused refund is stuck
forever. Deploys and container restarts happen all the time, so anything
that must outlive a request belongs in storage, not in a variable.

It's async (aiosqlite) for a reason: the checkpointer writes to the same
file from the same event loop. A blocking sqlite3 call waiting for the
file's lock would freeze the loop, so the checkpointer could never finish
and release the lock: a deadlock. Awaiting lets it finish.
"""

import json

import aiosqlite


class ApprovalQueue:
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    @classmethod
    async def open(cls, db_path: str) -> "ApprovalQueue":
        db = await aiosqlite.connect(db_path, timeout=10)
        await db.execute("""CREATE TABLE IF NOT EXISTS pending_approvals (
            thread_id TEXT PRIMARY KEY,
            customer  TEXT NOT NULL,
            actions   TEXT NOT NULL,
            created   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        await db.commit()
        return cls(db)

    async def add(self, thread_id: str, customer: str, actions: list[dict]) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO pending_approvals (thread_id, customer, actions) "
            "VALUES (?, ?, ?)", (thread_id, customer, json.dumps(actions)))
        await self._db.commit()

    async def take(self, thread_id: str) -> dict | None:
        """Remove and return one item. A single statement, so two staff
        members clicking at once can't both get it."""
        async with self._db.execute(
                "DELETE FROM pending_approvals WHERE thread_id = ? RETURNING customer, actions",
                (thread_id,)) as cursor:
            row = await cursor.fetchone()
        await self._db.commit()
        return {"customer": row[0], "actions": json.loads(row[1])} if row else None

    async def all(self) -> list[dict]:
        async with self._db.execute(
                "SELECT thread_id, customer, actions FROM pending_approvals ORDER BY created") as c:
            return [{"thread_id": t, "customer": cu, "actions": json.loads(a)}
                    async for t, cu, a in c]

    async def has(self, thread_id: str) -> bool:
        async with self._db.execute("SELECT 1 FROM pending_approvals WHERE thread_id = ?",
                                    (thread_id,)) as cursor:
            return await cursor.fetchone() is not None

    async def ping(self) -> None:
        """Raise if the database can't be read (used by /readyz)."""
        async with self._db.execute("SELECT 1") as cursor:
            await cursor.fetchone()

    async def close(self) -> None:
        await self._db.close()
