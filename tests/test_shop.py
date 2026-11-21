"""The business rules: the code that protects the money. No model involved."""

import asyncio

import pytest

from pedalwise import shop
from pedalwise.database import Database
from pedalwise.migrations import upgrade
from pedalwise.refunds import RefundLedger

SAM = shop.DEMO_CUSTOMERS["sam@example.com"]
DANA = shop.DEMO_CUSTOMERS["dana@example.com"]


def test_customers_only_see_their_own_orders():
    assert shop.find_order(SAM, "A1001") is not None
    assert shop.find_order(SAM, "a1001 ") is not None  # forgiving about format
    assert shop.find_order(DANA, "A1001") is None


@pytest.mark.parametrize("customer, order_id, reason", [
    (DANA, "A1001", "no such order"),          # someone else's order
    (SAM, "A1003", "hasn't been delivered"),    # still in transit
    (DANA, "A1004", "return window"),          # delivered 45 days ago
    (SAM, "A9999", "no such order"),           # doesn't exist
])
def test_refund_refusals(customer, order_id, reason):
    assert reason in shop.refund_problem(customer, order_id)


def test_eligible_until_refunded():
    assert shop.refund_problem(SAM, "A1001") is None
    assert shop.refund_amount(SAM, "A1001") == 899.0
    assert "already been refunded" in shop.refund_problem(SAM, "A1001", already_refunded=True)


def test_the_database_pays_each_order_once(db_url):
    """Twenty attempts at once, as if from several copies of the app: one wins."""
    async def scenario():
        db = await Database.connect(db_url)
        ledger = RefundLedger(db)
        await upgrade(db, log=lambda message: None)
        results = await asyncio.gather(*(ledger.record("a1001", SAM.email, 899.0)
                                          for _ in range(20)))
        assert results.count(True) == 1
        assert await ledger.is_refunded("A1001")
        assert await ledger.all() == [{"order_id": "A1001", "customer": SAM.email,
                                       "amount_eur": 899.0}]
        await db.close()
    asyncio.run(scenario())
