"""The business rules: the code that protects the money. No model involved."""

import pytest

from pedalwise import shop

SAM = shop.DEMO_CUSTOMERS["sam@example.com"]
DANA = shop.DEMO_CUSTOMERS["dana@example.com"]


@pytest.fixture(autouse=True)
def no_refunds():
    shop.REFUNDS.clear()


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


def test_eligible_refund_is_paid_once():
    assert shop.refund_problem(SAM, "A1001") is None
    assert shop.record_refund(SAM, "A1001") == 899.0
    assert "already been refunded" in shop.refund_problem(SAM, "A1001")
