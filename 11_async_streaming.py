"""
Tutorial 11 — Async and streaming events
========================================

Everything so far ran one step at a time and printed results at the end.
Web apps need two more things:
  - *async*: a server handles many users at once, so it must not sit idle
    while waiting for OpenAI to reply. Every LangChain method has an async
    twin: invoke -> ainvoke, stream -> astream, batch -> abatch.
  - *streaming events*: users want to see progress as it happens: tokens as
    they're written, "searching stock..." while a tool runs, and so on.

What you'll learn:
  1. Run calls concurrently with ainvoke + asyncio.gather
  2. Stream tokens with `async for` and astream()
  3. See every step inside a chain with astream_events()
  4. Stream a LangGraph agent: tokens, tool results and custom progress
  5. Wrap it all in one simple event generator, ready for a web server

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 11_async_streaming.py
  uv run 11_async_streaming.py "Do you have the City e-bike in stock in Porto?"
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Step 1: Concurrency with ainvoke + asyncio.gather
# ---------------------------------------------------------------------------
# `await llm.ainvoke(...)` pauses this function while waiting for the reply,
# letting other work run meanwhile. asyncio.gather starts several calls at
# once and waits for all of them, so 3 calls take about as long as 1.
async def concurrency_demo() -> None:
    print("=== Step 1: sequential vs concurrent ===")
    questions = ["Name a primary colour.", "Name a planet.", "Name an ocean."]

    start = time.perf_counter()
    for q in questions:
        await llm.ainvoke(q)  # each waits for the previous one to finish
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    replies = await asyncio.gather(*(llm.ainvoke(q) for q in questions))
    concurrent = time.perf_counter() - start

    print(f"sequential: {sequential:.1f}s   concurrent: {concurrent:.1f}s")
    print("answers:", [r.text.strip() for r in replies], "\n")
    # Tip: `await chain.abatch(inputs)` does the same for a list of inputs and
    # takes config={"max_concurrency": 5} to avoid hitting rate limits.


# ---------------------------------------------------------------------------
# Step 2: Stream tokens asynchronously
# ---------------------------------------------------------------------------
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You explain things in exactly three short sentences."),
        ("human", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)


async def token_stream_demo() -> None:
    print("=== Step 2: astream ===")
    async for piece in chain.astream({"topic": "Why is the sky blue?"}):
        print(piece, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# Step 3: astream_events — see inside the chain
# ---------------------------------------------------------------------------
# astream() only gives you the chain's *final* output, piece by piece.
# astream_events() reports what every component inside it is doing: when the
# prompt, the model and the parser start and end, plus each token. Each
# event is a dict with "event" (e.g. "on_chat_model_stream"), "name" and
# "data". version="v2" is the stable format (a typed "v3" is in beta).
async def events_demo() -> None:
    print("=== Step 3: astream_events ===")
    tokens = 0
    async for event in chain.astream_events({"topic": "What is a rainbow?"}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            tokens += 1  # one event per streamed chunk; too many to print
        elif kind.endswith(("_start", "_end")):
            print(f"  {kind:<22} {event['name']}")
    print(f"  ({tokens} on_chat_model_stream events in between)\n")
    # Filter with include_types=["chat_model"], include_names=[...] or tags
    # (chain.with_config(tags=["answer"])) to get only the events you need.


# ---------------------------------------------------------------------------
# Step 4: Streaming a LangGraph agent
# ---------------------------------------------------------------------------
# An async tool. get_stream_writer() lets the tool send its own progress
# messages to whoever is streaming the graph ("custom" stream mode).
STOCK = {("city e-bike", "porto"): 2, ("city e-bike", "lisbon"): 0,
         ("trail 500", "porto"): 5, ("trail 500", "lisbon"): 3}


@tool
async def check_stock(item: str, store: str) -> str:
    """Check how many units of an item a store has. Stores: Lisbon, Porto.
    Items: 'City e-bike', 'Trail 500'."""
    write = get_stream_writer()
    write(f"Contacting the {store} warehouse...")
    await asyncio.sleep(1)  # pretend this is a slow network call
    write(f"Counting units in {store}...")
    await asyncio.sleep(0.5)
    count = STOCK.get((item.strip().lower(), store.strip().lower()))
    return f"No data for {item} in {store}." if count is None else f"{count} x {item} in {store}"


TOOLS = [check_stock]
llm_with_tools = llm.bind_tools(TOOLS)
SYSTEM = SystemMessage("You are Pedalwise Bikes support. Use check_stock for "
                       "availability questions. Answer in one or two sentences.")


async def agent(state: MessagesState) -> dict:
    # An async node: LangGraph awaits it. Tokens are still streamed out to
    # the caller even though we use ainvoke here; LangGraph captures them.
    return {"messages": [await llm_with_tools.ainvoke([SYSTEM, *state["messages"]])]}


def build_app():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))  # runs async tools with await
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


# When the model asks for several tools at once (e.g. Porto AND Lisbon),
# ToolNode runs async tools concurrently, so their progress messages
# interleave. That's why each message names its store.
#
# With a list of stream modes, astream yields (mode, data) pairs:
#   "messages" -> (token chunk, metadata) as the model writes, from any node
#   "updates"  -> {node_name: state_update} after each node finishes
#   "custom"   -> whatever tools passed to get_stream_writer()
async def agent_stream_demo(app, question: str) -> None:
    print("=== Step 4: streaming the agent ===")
    print("customer:", question)
    inputs = {"messages": [("user", question)]}
    async for mode, data in app.astream(inputs, stream_mode=["messages", "updates", "custom"]):
        if mode == "messages":
            chunk, metadata = data
            # Only the agent's text; tool-call chunks have no text.
            if metadata["langgraph_node"] == "agent" and chunk.text:
                print(chunk.text, end="", flush=True)
        elif mode == "custom":
            print(f"  … {data}")
        elif mode == "updates":
            for node, update in data.items():
                last = update["messages"][-1]
                if node == "agent" and last.tool_calls:
                    for call in last.tool_calls:
                        print(f"  → calling {call['name']}({call['args']})")
                elif node == "tools":
                    for result in update["messages"]:
                        print(f"  ← {result.content}")
    print("\n")


# ---------------------------------------------------------------------------
# Step 5: One event stream for a web app
# ---------------------------------------------------------------------------
# A front end doesn't want LangGraph's internal shapes. Translate them into a
# few simple event types. Tutorial 12 sends exactly these over HTTP.
async def agent_events(app, question: str):
    """Yield {"type": "token" | "progress" | "tool_call" | "tool_result" | "done"}."""
    inputs = {"messages": [("user", question)]}
    async for mode, data in app.astream(inputs, stream_mode=["messages", "updates", "custom"]):
        if mode == "messages":
            chunk, metadata = data
            if (metadata["langgraph_node"] == "agent" and isinstance(chunk, AIMessageChunk)
                    and chunk.text):
                yield {"type": "token", "text": chunk.text}
        elif mode == "custom":
            yield {"type": "progress", "text": str(data)}
        elif mode == "updates":
            for node, update in data.items():
                if node == "agent":
                    for call in update["messages"][-1].tool_calls:
                        yield {"type": "tool_call", "name": call["name"], "args": call["args"]}
                elif node == "tools":
                    for result in update["messages"]:
                        yield {"type": "tool_result", "name": result.name, "text": result.content}
    yield {"type": "done"}


async def web_events_demo(app, question: str) -> None:
    print("=== Step 5: simplified events (what a web page would receive) ===")
    try:
        # A time limit matters on a server: never let one request hang forever.
        async with asyncio.timeout(60):
            async for event in agent_events(app, question):
                if event["type"] != "token":  # tokens are too many to list
                    print(" ", event)
    except TimeoutError:
        print("  gave up after 60 seconds")
    print()


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "Do you have the City e-bike in stock in Porto or Lisbon?"
    await concurrency_demo()
    await token_stream_demo()
    await events_demo()
    app = build_app()
    await agent_stream_demo(app, question)
    await web_events_demo(app, question)


if __name__ == "__main__":
    asyncio.run(main())  # the entry point that starts the async event loop
