"""
A small MCP server for Tutorial 14.

MCP (Model Context Protocol) is an open standard for offering tools and data
to AI applications. A server like this one doesn't know or care which model
or framework will use it: LangChain, Claude Desktop, an IDE and others can
all connect to the same server.

An MCP server can offer three kinds of things:
  - tools     functions the model can call (like @tool in Tutorial 05)
  - resources read-only data the app can fetch by URI (like files)
  - prompts   reusable prompt templates

You don't normally run this file yourself: Tutorial 14 starts it as a
subprocess and talks to it over stdin/stdout ("stdio" transport). To try it
on its own, run:  uv run mcp dev mcp_servers/pedalwise_server.py
(the MCP Inspector, a web UI for poking at servers; needs Node.js).
"""

import asyncio
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(__file__).parent.parent / "data"

# The name and instructions are sent to clients when they connect.
mcp = FastMCP(
    "pedalwise",
    instructions="Order lookup, stock checks and policy documents for Pedalwise Bikes.",
    log_level="WARNING",  # don't log every request to the client's terminal
)

ORDERS = {
    "A1001": {"item": "Trail 500 mountain bike", "price_eur": 899.0, "status": "delivered"},
    "A1002": {"item": "Commuter helmet", "price_eur": 59.0, "status": "delivered"},
    "A1003": {"item": "City e-bike", "price_eur": 2199.0, "status": "in transit, arriving Friday"},
}
STOCK = {("city e-bike", "porto"): 2, ("city e-bike", "lisbon"): 0,
         ("trail 500", "porto"): 5, ("trail 500", "lisbon"): 3}


# ---------------------------------------------------------------------------
# Tools: FastMCP reads the name, type hints and docstring, exactly like
# LangChain's @tool, and publishes them as a schema clients can discover.
# ---------------------------------------------------------------------------
@mcp.tool()
def get_order(order_id: str) -> str:
    """Look up an order by ID (e.g. 'A1001'): item, price and delivery status."""
    order = ORDERS.get(order_id.strip().upper())
    return str(order) if order else f"No order with ID {order_id}."


@mcp.tool()
async def check_stock(item: str, store: str) -> str:
    """Check how many units of an item a store has. Stores: Lisbon, Porto.
    Items: 'City e-bike', 'Trail 500'."""
    await asyncio.sleep(0.5)  # pretend this is a slow warehouse system
    count = STOCK.get((item.strip().lower(), store.strip().lower()))
    return f"No data for {item} in {store}." if count is None else f"{count} x {item} in {store}"


# ---------------------------------------------------------------------------
# Resources: data addressed by a URI. "{name}" makes this a template, so
# pedalwise://policies/warranty serves data/warranty.md.
# ---------------------------------------------------------------------------
@mcp.resource("pedalwise://policies/{name}", mime_type="text/markdown")
def policy(name: str) -> str:
    """A Pedalwise policy document: returns_policy, warranty or servicing."""
    path = DATA_DIR / f"{name}.md"
    # Only serve files that really are in data/: never let a URI like
    # "../../secret" read files elsewhere on the machine.
    if path.resolve().parent != DATA_DIR.resolve() or not path.is_file():
        raise ValueError(f"Unknown policy: {name}")
    return path.read_text(encoding="utf-8")


@mcp.resource("pedalwise://opening-hours")
def opening_hours() -> str:
    """Workshop opening hours."""
    return "Monday to Saturday, 9:00 to 18:00. Closed Sundays and public holidays."


# ---------------------------------------------------------------------------
# Prompts: reusable templates a client can fill in and send to its model.
# ---------------------------------------------------------------------------
@mcp.prompt()
def apology_email(customer_name: str, problem: str) -> str:
    """Draft a short apology email to a customer."""
    return (f"Write a short, warm apology email from Pedalwise Bikes to {customer_name} "
            f"about this problem: {problem}. Offer a concrete next step. Sign it 'The "
            f"Pedalwise team'.")


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
