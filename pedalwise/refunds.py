"""The refund ledger: which orders have been refunded (Tutorial 26).

Tutorial 20 kept this as a Python list. That had two bugs nobody noticed
while one process ran for a short demo:
  - a restart emptied the list, so a refunded order became refundable again;
  - with two copies of the app, each had its own list, so both could pay.

Now it's a table whose primary key is the order id. The database itself
refuses a second row for the same order, so even two copies approving the
same refund at the same instant pay exactly once. "Check, then write" in
Python can't promise that: both copies could check before either writes.
"""

from .database import Database, now_text


class RefundLedger:
    def __init__(self, db: Database):
        self._db = db

    async def setup(self) -> None:
        await self._db.execute("""CREATE TABLE IF NOT EXISTS refunds (
            order_id   TEXT PRIMARY KEY,
            customer   TEXT NOT NULL,
            amount_eur DOUBLE PRECISION NOT NULL,
            created    TEXT NOT NULL)""")

    async def is_refunded(self, order_id: str) -> bool:
        return await self._db.fetchone("SELECT 1 FROM refunds WHERE order_id = ?",
                                       (order_id.strip().upper(),)) is not None

    async def record(self, order_id: str, customer: str, amount_eur: float) -> bool:
        """Record a refund. False if this order was already refunded."""
        row = await self._db.fetchone(
            "INSERT INTO refunds (order_id, customer, amount_eur, created) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (order_id) DO NOTHING RETURNING order_id",
            (order_id.strip().upper(), customer, amount_eur, now_text()))
        return row is not None

    async def all(self) -> list[dict]:
        rows = await self._db.fetchall(
            "SELECT order_id, customer, amount_eur FROM refunds ORDER BY created, order_id")
        return [{"order_id": o, "customer": c, "amount_eur": a} for o, c, a in rows]
