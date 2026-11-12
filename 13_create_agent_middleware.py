"""
Tutorial 13 — create_agent and middleware
=========================================

Tutorials 05-12 built agents by hand so you could see how they work: the
model node, the tool node, the loop, the checkpointer. For everyday work,
LangChain packages all of that into one function, create_agent(), and lets
you add behaviour with *middleware*: small plug-ins that run before or
after each model call or around each tool call.

What you'll learn:
  1. Build Tutorial 07's agent in one call with create_agent()
  2. Get typed results from an agent with response_format
  3. Add built-in middleware: limits, PII redaction, summarization
  4. Get Tutorial 08's human approval with one middleware
  5. Write your own middleware: a per-user prompt and a tool-call logger

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 13_create_agent_middleware.py
"""

import os
import time
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)

# ---------------------------------------------------------------------------
# Tools (from Tutorials 08 and 12)
# ---------------------------------------------------------------------------
ORDERS = {
    "A1001": {"item": "Trail 500 mountain bike", "price_eur": 899.0, "status": "delivered"},
    "A1002": {"item": "Commuter helmet", "price_eur": 59.0, "status": "delivered"},
    "A1003": {"item": "City e-bike", "price_eur": 2199.0, "status": "in transit, arriving Friday"},
}
REFUNDS_ISSUED: list[dict] = []


@tool
def get_order(order_id: str) -> str:
    """Look up an order by ID (e.g. 'A1001'): item, price and status."""
    order = ORDERS.get(order_id.strip().upper())
    return str(order) if order else f"No order with ID {order_id}."


@tool
def issue_refund(order_id: str, amount_eur: float) -> str:
    """Refund money for an order. Look up the order first for the price."""
    REFUNDS_ISSUED.append({"order_id": order_id, "amount_eur": amount_eur})
    return f"Refund of {amount_eur:.2f} EUR issued for order {order_id}."


SYSTEM_PROMPT = ("You are Pedalwise Bikes support. Use the tools to look up orders "
                 "and issue refunds. Answer in one to three short sentences.")


def ask(agent, text: str, thread_id: str = "demo", **kwargs) -> dict:
    """Send one user message and print the agent's reply."""
    print(f"customer: {text}")
    result = agent.invoke({"messages": [{"role": "user", "content": text}]},
                          {"configurable": {"thread_id": thread_id}}, **kwargs)
    if "__interrupt__" not in result:
        print(f"assistant: {result['messages'][-1].text}")
    return result


# ---------------------------------------------------------------------------
# Step 1: create_agent replaces the hand-built graph
# ---------------------------------------------------------------------------
# One call gives you what Tutorial 07 wrote in ~25 lines: a model node, a
# tool node, the loop between them, and memory via the checkpointer. The
# result is a normal LangGraph graph, so invoke/stream/astream all work.
def basic_agent_demo() -> None:
    print("=== Step 1: create_agent ===")
    agent = create_agent(
        llm,
        tools=[get_order],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    ask(agent, "Where is my order A1003?")
    ask(agent, "And what did it cost?")  # same thread: it remembers A1003
    print()


# ---------------------------------------------------------------------------
# Step 2: Typed results with response_format
# ---------------------------------------------------------------------------
# Like .with_structured_output() in Tutorial 03, but for a whole agent run:
# the agent uses its tools as usual, then fills in this model at the end.
class TicketSummary(BaseModel):
    """A support ticket created from the conversation."""

    order_id: str | None = Field(description="The order discussed, if any")
    category: Literal["delivery", "refund", "product question", "other"]
    summary: str = Field(description="One sentence describing the customer's issue")
    needs_human: bool = Field(description="True if staff must follow up")


def structured_agent_demo() -> None:
    print("=== Step 2: response_format ===")
    agent = create_agent(llm, tools=[get_order], system_prompt=SYSTEM_PROMPT,
                         response_format=TicketSummary)
    result = ask(agent, "Order A1001 arrived with a scratched frame. Not happy!")
    ticket: TicketSummary = result["structured_response"]
    print("ticket:", ticket.model_dump(), "\n")


# ---------------------------------------------------------------------------
# Step 3: Built-in middleware
# ---------------------------------------------------------------------------
# Middleware is a list; each item hooks into the agent loop. Order matters:
# they run in the order listed before a model call, reversed after it.
def builtin_middleware_demo() -> None:
    print("=== Step 3: built-in middleware ===")
    agent = create_agent(
        llm,
        tools=[get_order],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            # Hide personal data before it reaches the model (and its logs).
            # Strategies: "redact" -> [REDACTED_EMAIL], "mask" keeps the
            # last digits, "hash", or "block" to refuse the message.
            PIIMiddleware("email", strategy="redact"),
            PIIMiddleware("credit_card", strategy="mask"),
            # Safety nets, like Tutorial 09's step limit: stop a confused
            # agent from looping and running up your bill.
            ModelCallLimitMiddleware(run_limit=6),
            ToolCallLimitMiddleware(tool_name="get_order", run_limit=3),
            # When the conversation passes 10 messages, replace the oldest
            # with a summary, keeping the 4 most recent as they are. This is
            # Tutorial 04's trimming, but the old turns aren't simply lost.
            SummarizationMiddleware(model=llm, trigger=("messages", 10), keep=("messages", 4)),
        ],
    )
    result = ask(agent, "Hi, I'm Sam (sam@example.com), card 4111 1111 1111 1111. "
                        "Where is order A1003?", thread_id="sam")
    # What the model actually received:
    print("stored user message:", result["messages"][0].text)

    for text in ["When will it arrive?", "What colour options does it come in?",
                 "Can I collect it from the shop instead?", "What's my name?"]:
        result = ask(agent, text, thread_id="sam")
    print(f"(thread holds {len(result['messages'])} messages after summarizing)\n")


# ---------------------------------------------------------------------------
# Step 4: Human approval as middleware
# ---------------------------------------------------------------------------
# Tutorial 08 put interrupt() inside the tool. HumanInTheLoopMiddleware does
# it for you: list the tools that need approval, and the tool itself stays a
# plain function. Like Tutorial 08, this needs a checkpointer.
def approval_demo() -> None:
    print("=== Step 4: HumanInTheLoopMiddleware ===")
    agent = create_agent(
        llm,
        tools=[get_order, issue_refund],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={
                "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
                "get_order": False,  # safe: runs without asking
            },
            interrupt_mode="batched",  # one pause per model turn, listing every call
        )],
    )
    config = {"configurable": {"thread_id": "refund"}}
    result = ask(agent, "Order A1002 broke on day one. Please refund it.", thread_id="refund")

    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        for action in request["action_requests"]:
            print(f"  ⏸  approval needed: {action['name']}({action['args']})")
        # One decision per action, in order. Options: {"type": "approve"},
        # {"type": "edit", "edited_action": {"name": ..., "args": {...}}},
        # {"type": "reject", "message": "why"}.
        decisions = [{"type": "approve"} for _ in request["action_requests"]]
        print(f"  ▶  reviewer: {decisions}")
        result = agent.invoke(Command(resume={"decisions": decisions}), config)

    print(f"assistant: {result['messages'][-1].text}")
    print("refunds issued:", REFUNDS_ISSUED, "\n")


# ---------------------------------------------------------------------------
# Step 5: Your own middleware
# ---------------------------------------------------------------------------
# Runtime context: per-request data that isn't part of the conversation,
# such as who is logged in. Declare its shape with context_schema and pass it
# with agent.invoke(..., context=...). Middleware and tools can read it.
@dataclass
class Customer:
    name: str
    tier: Literal["standard", "gold"]


# @dynamic_prompt builds the system prompt fresh for every model call.
@dynamic_prompt
def personalised_prompt(request: ModelRequest) -> str:
    customer: Customer = request.runtime.context
    perks = ("Gold members get free express servicing; mention it when relevant."
             if customer.tier == "gold" else "")
    return f"{SYSTEM_PROMPT} The customer's name is {customer.name}. {perks}"


# @wrap_tool_call runs around every tool call. `handler(request)` runs the
# tool; anything before or after it is yours: logging, timing, retries,
# permission checks, changing the result.
@wrap_tool_call
def log_tool_calls(request, handler):
    call = request.tool_call
    start = time.perf_counter()
    result = handler(request)
    print(f"  [log] {call['name']}({call['args']}) took "
          f"{(time.perf_counter() - start) * 1000:.0f} ms")
    return result


def custom_middleware_demo() -> None:
    print("=== Step 5: custom middleware ===")
    agent = create_agent(
        llm,
        tools=[get_order],
        context_schema=Customer,
        middleware=[personalised_prompt, log_tool_calls],
    )
    for customer in [Customer("Alex", "gold"), Customer("Jordan", "standard")]:
        print(f"-- as {customer.name} ({customer.tier}) --")
        ask(agent, "My order A1001 arrived. Any tips for getting the bike serviced?",
            thread_id=customer.name, context=customer)
    print()


if __name__ == "__main__":
    basic_agent_demo()
    structured_agent_demo()
    builtin_middleware_demo()
    approval_demo()
    custom_middleware_demo()
