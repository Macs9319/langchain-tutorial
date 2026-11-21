"""One small async database interface over SQLite and Postgres (Tutorial 26).

SQLite is a file on one machine: perfect for one copy of the app. Several
copies behind a load balancer need one database they all share, which is
Postgres here. The app's own tables (approval queue, refund ledger,
telemetry) go through this class, so the same code runs on both:

    db = await Database.connect("pedalwise.sqlite")                # local
    db = await Database.connect("postgresql://user:pw@host/db")    # shared

The SQL is written once, in the subset both understand: ? placeholders
(rewritten to %s for Postgres), ON CONFLICT for upserts, RETURNING, and
timestamps as "YYYY-MM-DD HH:MM:SS" text computed in Python, so no
database-specific date functions are needed.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import aiosqlite

# Any fixed number: every copy of the app uses it to take the same lock.
SCHEMA_LOCK_ID = 2026_11_21


def is_postgres_url(url: str) -> bool:
    return url.startswith(("postgresql://", "postgres://"))


def now_text(minus: timedelta = timedelta()) -> str:
    """UTC time as sortable text, the same format as SQLite's CURRENT_TIMESTAMP."""
    return (datetime.now(UTC) - minus).strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, url: str):
        self.url = url
        self.postgres = is_postgres_url(url)
        self._sqlite: aiosqlite.Connection | None = None
        self._pool = None

    @classmethod
    async def connect(cls, url: str) -> "Database":
        db = cls(url)
        if db.postgres:
            from psycopg_pool import AsyncConnectionPool
            # A pool: several requests can use the database at once, each on
            # its own connection, instead of queuing for a single one.
            db._pool = AsyncConnectionPool(url, min_size=1, max_size=10, open=False)
            await db._pool.open(wait=True, timeout=30)
        else:
            db._sqlite = await aiosqlite.connect(url, timeout=10)
        return db

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.postgres else sql

    async def execute(self, sql: str, params: tuple = ()) -> None:
        if self.postgres:
            async with self._pool.connection() as conn:  # commits when the block ends
                await conn.execute(self._sql(sql), params)
        else:
            await self._sqlite.execute(sql, params)
            await self._sqlite.commit()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Rows from a query. Also commits, so it works for DELETE ... RETURNING."""
        if self.postgres:
            async with self._pool.connection() as conn:
                cursor = await conn.execute(self._sql(sql), params)
                return await cursor.fetchall() if cursor.description else []
        async with self._sqlite.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        await self._sqlite.commit()
        return [tuple(row) for row in rows]

    async def fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        rows = await self.fetchall(sql, params)
        return rows[0] if rows else None

    @asynccontextmanager
    async def schema_lock(self) -> AsyncIterator[None]:
        """Only one copy of the app creates tables at a time.

        Two copies starting together both run CREATE TABLE IF NOT EXISTS; in
        Postgres that can collide and crash one of them. An advisory lock is
        a named lock in the database: the second copy waits until the first
        has finished. (Bigger apps run migrations as a separate job instead.)
        SQLite has one writer at a time anyway, so it needs nothing.
        """
        if not self.postgres:
            yield
            return
        import psycopg
        # Two traps here, both found the hard way. LangGraph's setup builds
        # indexes with CREATE INDEX CONCURRENTLY, which waits until every
        # transaction and statement running anywhere in the database is done:
        #  1. So the lock is held on its own autocommit connection: a lock
        #     taken inside an open transaction would wait for itself.
        #  2. And a copy waiting for the lock must not wait *inside* a
        #     statement (pg_advisory_lock blocks in one): the index build in
        #     the copy holding the lock would wait for that statement, which
        #     waits for the lock. Postgres can't see that cycle. So: try the
        #     lock, and if it's taken, sleep with no statement running.
        async with await psycopg.AsyncConnection.connect(self.url, autocommit=True) as conn:
            while not (await (await conn.execute(
                    "SELECT pg_try_advisory_lock(%s)", (SCHEMA_LOCK_ID,))).fetchone())[0]:
                await asyncio.sleep(0.2)
            try:
                yield
            finally:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (SCHEMA_LOCK_ID,))

    async def close(self) -> None:
        if self.postgres:
            await self._pool.close()
        else:
            await self._sqlite.close()
