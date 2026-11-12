"""
Tutorial 09 — Multi-agent: a supervisor and specialists
=======================================================

One agent with every tool works for small jobs, but as tools and
instructions pile up, it gets confused and its prompt gets long. A common
fix is to split the work: small *specialist* agents that each do one thing
well, and a *supervisor* that decides who handles what.

Pedalwise Bikes support, split into:
  - orders   : looks up orders and delivery status
  - policies : searches the returns / warranty / servicing documents
  - supervisor: reads the conversation, picks the next specialist or finishes
  - respond  : writes one friendly reply from everything the specialists found

What you'll learn:
  1. Build each specialist as its own small LangGraph agent
  2. Make a supervisor that routes with structured output
  3. Wire them into one graph with conditional edges
  4. Keep the main conversation clean (specialists return only conclusions)
  5. Guard against endless loops with a step limit in the state

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 09_multi_agent.py
  uv run 09_multi_agent.py "Where is order A1003, and how long is the e-bike motor warranty?"
"""

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv()

HERE = Path(__file__).parent
MAX_STEPS = 6  # supervisor decisions allowed per customer message

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Tools (from Tutorials 06 and 08)
# ---------------------------------------------------------------------------
ORDERS = {
    "A1001": {"item": "Trail 500 mountain bike", "price_eur": 899.0, "status": "delivered 2 days ago"},
    "A1002": {"item": "Commuter helmet", "price_eur": 59.0, "status": "delivered last month"},
    "A1003": {"item": "City e-bike", "price_eur": 2199.0, "status": "in transit, arriving Friday"},
}


@tool
def get_order(order_id: str) -> str:
    """Look up an order by ID (e.g. 'A1001'): item, price and delivery status."""
    order = ORDERS.get(order_id.strip().upper())
    return str(order) if order else f"No order with ID {order_id}."


_store: InMemoryVectorStore | None = None


@tool
def search_docs(query: str) -> str:
    """Search Pedalwise Bikes' policies: returns, refunds, warranty, servicing,
    prices and opening hours."""
    global _store
    if _store is None:  # build the index on first use, as in Tutorial 07
        docs = [Document(p.read_text(encoding="utf-8"), metadata={"source": p.name})
                for p in sorted((HERE / "data").glob("*.md"))]
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
        _store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
        _store.add_documents(chunks)
    results = _store.similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in results)


# ---------------------------------------------------------------------------
# Step 1: Specialists, each a small agent graph
# ---------------------------------------------------------------------------
# This is the Tutorial 07 graph, made reusable. Each specialist gets only its
# own tools and a narrow instruction, which keeps it focused and reliable.
def make_specialist(tools: list, instructions: str):
    model = llm.bind_tools(tools)
    system = SystemMessage(instructions + " Reply with a short factual summary of "
                           "what you found, for a colleague (not the customer).")

    def call_model(state: MessagesState) -> dict:
        return {"messages": [model.invoke([system, *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


SPECIALISTS = {
    "orders": make_specialist(
        [get_order], "You are the orders specialist. Look up order details and status."),
    "policies": make_specialist(
        [search_docs], "You are the policies specialist. Find the relevant shop policy "
                       "and quote the exact numbers (days, prices, years)."),
}


# ---------------------------------------------------------------------------
# Step 4: Specialists as nodes in the main graph
# ---------------------------------------------------------------------------
# The specialist's tool calls and raw search results stay inside its own run.
# Only its final summary goes into the shared conversation, labelled with the
# specialist's name. That keeps the supervisor's context short and readable.
def specialist_node(name: str):
    def run(state: "SupportState") -> dict:
        result = SPECIALISTS[name].invoke({"messages": state["messages"]})
        summary = result["messages"][-1].text
        print(f"  [{name}] {summary}")
        return {"messages": [AIMessage(summary, name=name)]}
    return run


# ---------------------------------------------------------------------------
# Step 2: The supervisor routes with structured output
# ---------------------------------------------------------------------------
# Instead of free text, the supervisor fills in a Route object (Tutorial 03),
# so its decision is always one of the allowed node names.
class Route(BaseModel):
    """Who should act next."""

    reason: str = Field(description="One short sentence explaining the choice")
    next: Literal["orders", "policies", "respond"] = Field(
        description="'orders' or 'policies' if information is still missing; "
                    "'respond' once everything needed to answer is known")


router = llm.with_structured_output(Route)

SUPERVISOR_PROMPT = SystemMessage(
    "You supervise Pedalwise Bikes support. Specialists: 'orders' (order details "
    "and delivery status) and 'policies' (returns, warranty, servicing, prices, "
    "hours). Read the conversation, including what specialists already reported, "
    "and pick who acts next. Don't ask a specialist for something it already "
    "answered. Choose 'respond' when every part of the customer's question is covered."
)


# Step 5: the state gets an extra key beyond "messages". Plain keys like these
# are overwritten by node updates (only "messages" is appended to).
class SupportState(MessagesState):
    next: str
    steps: int


def supervisor(state: SupportState) -> dict:
    steps = state.get("steps", 0) + 1
    if steps > MAX_STEPS:  # safety net: never bounce between agents forever
        print("  [supervisor] step limit reached -> respond")
        return {"next": "respond", "steps": steps}
    route = router.invoke([SUPERVISOR_PROMPT, *state["messages"]])
    print(f"  [supervisor] -> {route.next}: {route.reason}")
    return {"next": route.next, "steps": steps}


def respond(state: SupportState) -> dict:
    """Write the customer-facing answer from the specialists' findings."""
    reply = llm.invoke([
        SystemMessage("You are Pedalwise Bikes support. Using the specialists' notes "
                      "in the conversation, write one short, friendly reply to the "
                      "customer that answers every part of their question."),
        *state["messages"],
    ])
    return {"messages": [reply]}


# ---------------------------------------------------------------------------
# Step 3: Wire everything together
# ---------------------------------------------------------------------------
#   START -> supervisor --orders---> orders ---+
#                ^      --policies-> policies -+--> back to supervisor
#                |      --respond--> respond -> END
#                +----------------------------+
def build_app():
    graph = StateGraph(SupportState)
    graph.add_node("supervisor", supervisor)
    for name in SPECIALISTS:
        graph.add_node(name, specialist_node(name))
        graph.add_edge(name, "supervisor")  # report back after each job
    graph.add_node("respond", respond)

    graph.add_edge(START, "supervisor")
    # Route on the supervisor's decision stored in state["next"].
    graph.add_conditional_edges("supervisor", lambda state: state["next"],
                                ["orders", "policies", "respond"])
    graph.add_edge("respond", END)
    return graph.compile()


def ask(app, question: str) -> None:
    print(f"customer: {question}")
    result = app.invoke({"messages": [("user", question)], "steps": 0})
    print(f"assistant: {result['messages'][-1].text}\n")


if __name__ == "__main__":
    app = build_app()
    if len(sys.argv) > 1:
        ask(app, " ".join(sys.argv[1:]))
    else:
        print("=== Graph (Mermaid, paste into mermaid.live) ===")
        print(app.get_graph().draw_mermaid())
        ask(app, "Where is my order A1003?")                               # orders only
        ask(app, "Are you open on Saturdays?")                             # policies only
        ask(app, "My order A1001 arrived and I've ridden it once, but it's "
                 "too small. Can I still return it, and what would it cost me?")  # both
