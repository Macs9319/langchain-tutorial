"""What the agent can do.

Every tool reads the customer from runtime context, never from the model's
arguments, so a customer can only ever see and act on their own orders.
"""

import asyncio

from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer

from . import shop
from .knowledge import KnowledgeBase


def make_tools(kb: KnowledgeBase) -> list:
    @tool
    def search_policies(query: str) -> str:
        """Search Pedalwise's policies and product catalog: returns, warranty,
        servicing, opening hours, bikes, spare parts and part numbers."""
        results = kb.search(query, k=4)
        return "\n\n".join(d.page_content for d in results) or "Nothing found."

    @tool
    def my_orders(runtime: ToolRuntime[shop.Customer]) -> str:
        """List the current customer's orders with their status."""
        orders = {oid: o for oid, o in shop.ORDERS.items()
                  if o["customer"] == runtime.context.email}
        return "\n".join(f"{oid}: {o['item']}, {o['status']}, {o['price_eur']:.2f} EUR"
                         for oid, o in orders.items()) or "No orders."

    @tool
    async def check_stock(item: str, store: str) -> str:
        """Check stock of a bike at a store. Stores: Lisbon, Porto.
        Items: 'City e-bike', 'Trail 500', 'Gravel 300'."""
        get_stream_writer()(f"Checking the {store} warehouse…")
        await asyncio.sleep(0.5)  # pretend this is a slow warehouse system
        count = shop.stock_level(item, store)
        return f"No data for {item} in {store}." if count is None else f"{count} x {item} in {store}"

    @tool
    def check_refund_eligibility(order_id: str, runtime: ToolRuntime[shop.Customer]) -> str:
        """Check whether an order can be refunded. Always do this before
        request_refund, and explain the reason if it can't."""
        problem = shop.refund_problem(runtime.context, order_id)
        return f"Not eligible: {problem}." if problem else "Eligible for a full refund."

    @tool
    def request_refund(order_id: str, reason: str, runtime: ToolRuntime[shop.Customer]) -> str:
        """Refund an eligible order in full. A staff member must approve it
        first; the conversation pauses until they decide."""
        customer = runtime.context
        # Checked again here: this is the rule that actually protects the money,
        # whatever happened earlier in the conversation.
        problem = shop.refund_problem(customer, order_id)
        if problem:
            return f"Refund refused: {problem}."
        amount = shop.record_refund(customer, order_id)
        return f"Refund of {amount:.2f} EUR approved and issued for order {order_id.upper()}."

    return [search_policies, my_orders, check_stock, check_refund_eligibility, request_refund]
