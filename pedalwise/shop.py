"""The shop's data and business rules.

Nothing here knows about LLMs. The rules are plain Python, so they hold no
matter what the model is told or tricked into asking (Tutorial 18), and
they can be tested without any API calls (tests/test_shop.py).
"""

from dataclasses import dataclass

# Fake data standing in for a real database. All of it is made up.
ORDERS = {
    "A1001": {"customer": "sam@example.com", "item": "Trail 500 mountain bike",
              "price_eur": 899.0, "status": "delivered", "delivered_days_ago": 2},
    "A1002": {"customer": "dana@example.com", "item": "Commuter helmet",
              "price_eur": 59.0, "status": "delivered", "delivered_days_ago": 9},
    "A1003": {"customer": "sam@example.com", "item": "City e-bike",
              "price_eur": 2199.0, "status": "in transit, arriving Friday",
              "delivered_days_ago": None},
    "A1004": {"customer": "dana@example.com", "item": "Gravel 300",
              "price_eur": 1099.0, "status": "delivered", "delivered_days_ago": 45},
}
STOCK = {("city e-bike", "porto"): 2, ("city e-bike", "lisbon"): 0,
         ("trail 500", "porto"): 5, ("trail 500", "lisbon"): 3,
         ("gravel 300", "porto"): 1, ("gravel 300", "lisbon"): 4}
REFUNDS: list[dict] = []

RETURN_WINDOW_DAYS = 30  # matches data/returns_policy.md


@dataclass(frozen=True)
class Customer:
    """Who is chatting. In a real app this comes from their login session."""

    email: str
    name: str


DEMO_CUSTOMERS = {
    "sam@example.com": Customer("sam@example.com", "Sam"),
    "dana@example.com": Customer("dana@example.com", "Dana"),
}


def find_order(customer: Customer, order_id: str) -> dict | None:
    """An order, but only if it belongs to this customer."""
    order = ORDERS.get(order_id.strip().upper())
    return order if order and order["customer"] == customer.email else None


def refund_problem(customer: Customer, order_id: str) -> str | None:
    """Why this refund must be refused, or None if it may go ahead."""
    order = find_order(customer, order_id)
    if order is None:
        return "no such order on this customer's account"
    if order["delivered_days_ago"] is None:
        return "the order hasn't been delivered yet"
    if order["delivered_days_ago"] > RETURN_WINDOW_DAYS:
        return f"the {RETURN_WINDOW_DAYS}-day return window has passed"
    if any(r["order_id"] == order_id.upper() for r in REFUNDS):
        return "this order has already been refunded"
    return None


def record_refund(customer: Customer, order_id: str) -> float:
    """Refund the full price. Call only after refund_problem() returned None."""
    order = find_order(customer, order_id)
    REFUNDS.append({"order_id": order_id.upper(), "customer": customer.email,
                    "amount_eur": order["price_eur"]})
    return order["price_eur"]


def stock_level(item: str, store: str) -> int | None:
    return STOCK.get((item.strip().lower(), store.strip().lower()))
