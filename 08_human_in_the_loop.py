"""
Tutorial 08 — Human in the loop
===============================

Some actions shouldn't happen just because a model decided they should:
refunds, payments, deleting data, sending emails. LangGraph can *pause* a
graph in the middle of a run, wait (seconds or days) for a person to decide,
and then continue exactly where it stopped.

The agent is a Pedalwise Bikes support assistant. Looking up an order is
safe, so it runs freely. Issuing a refund needs a staff member's approval.

What you'll learn:
  1. Pause inside a tool with interrupt()
  2. Detect the pause: the result contains "__interrupt__"
  3. Resume with Command(resume=...) and approve the action
  4. Let the reviewer edit or reject the action instead
  5. Approve actions interactively (--chat)

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 08_human_in_the_loop.py          # scripted demos: approve, edit, reject
  uv run 08_human_in_the_loop.py --chat   # you play customer AND staff reviewer
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)

# Fake order database and a log of refunds actually issued.
ORDERS = {
    "A1001": {"item": "Trail 500 mountain bike", "price_eur": 899.0, "status": "delivered"},
    "A1002": {"item": "Commuter helmet", "price_eur": 59.0, "status": "delivered"},
    "A1003": {"item": "City e-bike", "price_eur": 2199.0, "status": "in transit"},
}
REFUNDS_ISSUED: list[dict] = []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_order(order_id: str) -> str:
    """Look up an order by its ID (e.g. 'A1001'): item, price and status."""
    order = ORDERS.get(order_id.strip().upper())
    return str(order) if order else f"No order with ID {order_id}."


# ---------------------------------------------------------------------------
# Step 1: interrupt() inside a sensitive tool
# ---------------------------------------------------------------------------
# interrupt(payload) stops the whole graph and saves its state with the
# checkpointer. `payload` is whatever the reviewer needs to see. When the
# graph is resumed, interrupt() *returns* the reviewer's decision and the
# function carries on from that line.
#
# Important: on resume, LangGraph re-runs this function from the top, so any
# side effect (charging a card, sending an email) must come AFTER interrupt().
@tool
def issue_refund(order_id: str, amount_eur: float, reason: str) -> str:
    """Refund money to a customer. Requires staff approval, which happens
    automatically. Look up the order first to get the correct price."""
    decision = interrupt({
        "action": "issue_refund",
        "order_id": order_id,
        "amount_eur": amount_eur,
        "reason": reason,
    })

    if decision["type"] == "reject":
        return f"Refund NOT issued. Staff rejected it: {decision.get('note', 'no reason given')}"
    if decision["type"] == "edit":
        amount_eur = decision["amount_eur"]  # the reviewer changed the amount

    # Only reached after approval: now perform the real action.
    REFUNDS_ISSUED.append({"order_id": order_id, "amount_eur": amount_eur})
    return f"Refund of {amount_eur:.2f} EUR issued for order {order_id}."


TOOLS = [get_order, issue_refund]
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM = SystemMessage(
    "You are the support assistant for Pedalwise Bikes. Use get_order to check "
    "orders. When a refund is justified, call issue_refund; staff review every "
    "refund. Tell the customer the outcome briefly."
)


def agent(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke([SYSTEM, *state["messages"]])]}


def build_app():
    """Same graph shape as Tutorial 07. A checkpointer is required: it is
    where the paused state waits until someone resumes it."""
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Step 2 & 3: Run until paused, then resume
# ---------------------------------------------------------------------------
def run_with_review(app, thread_id: str, text: str, review, echo: bool = True) -> None:
    """Run one customer message, handing every pause to `review`.

    `review` is a function: interrupt payload in, decision dict out. In the
    demos it's scripted; in --chat it asks you at the keyboard. In a real app
    the pause could last days: the decision would arrive later from a staff
    dashboard, and you'd resume with the same thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    if echo:
        print(f"customer: {text}")
    result = app.invoke({"messages": [("user", text)]}, config)

    # Step 2: a paused run returns its pending interrupts under "__interrupt__".
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"  ⏸  paused for approval: {request}")
        # The paused state is saved; .next shows which node will resume.
        print(f"     waiting at node: {app.get_state(config).next}")

        decision = review(request)
        print(f"  ▶  reviewer decided: {decision}")

        # Step 3: resume. The value becomes interrupt()'s return value.
        result = app.invoke(Command(resume=decision), config)

    print(f"assistant: {result['messages'][-1].text}\n")


def demos() -> None:
    app = build_app()
    complaint = "My order A1001 arrived with a cracked frame. I want a full refund."

    print("=== Step 3: approve ===")
    run_with_review(app, "t-approve", complaint, lambda request: {"type": "approve"})

    # -----------------------------------------------------------------------
    # Step 4: edit or reject
    # -----------------------------------------------------------------------
    print("=== Step 4a: edit (reviewer offers a partial refund) ===")
    run_with_review(app, "t-edit", complaint,
                    lambda request: {"type": "edit", "amount_eur": request["amount_eur"] * 0.5})

    print("=== Step 4b: reject ===")
    run_with_review(app, "t-reject",
                    "Refund my order A1002 please, I just don't like the colour.",
                    lambda request: {"type": "reject",
                                     "note": "colour preference: offer an exchange instead"})

    print("refunds actually issued:", REFUNDS_ISSUED)


# ---------------------------------------------------------------------------
# Step 5: Interactive review
# ---------------------------------------------------------------------------
def keyboard_review(request: dict) -> dict:
    print(f"\n  STAFF REVIEW — refund {request['amount_eur']:.2f} EUR "
          f"for {request['order_id']} ({request['reason']})")
    while True:
        choice = input("  [a]pprove, [e]dit amount, [r]eject? ").strip().lower()
        if choice == "a":
            return {"type": "approve"}
        if choice == "e":
            try:
                return {"type": "edit", "amount_eur": float(input("  new amount (EUR): "))}
            except ValueError:
                print("  please enter a number")
        if choice == "r":
            return {"type": "reject", "note": input("  reason: ").strip() or "not approved"}


def chat() -> None:
    app = build_app()
    print("You are the customer; when a refund is requested you are also the "
          "staff reviewer.\nOrders: " + ", ".join(ORDERS) + ". Type 'quit' to exit.\n")
    while True:
        try:
            text = input("customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            run_with_review(app, "cli", text, keyboard_review, echo=False)
    print("refunds issued this session:", REFUNDS_ISSUED)


if __name__ == "__main__":
    if "--chat" in sys.argv:
        chat()
    else:
        demos()
