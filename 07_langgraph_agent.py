"""
Tutorial 07 — An agent with LangGraph
=====================================

In Tutorial 05 we wrote the tool loop by hand, and in Tutorial 04 we kept
memory in a Python list that vanished when the script exited. LangGraph
replaces both: you describe the agent as a *graph* of steps, and a
*checkpointer* saves the conversation after every step, even to disk.

The agent here is a support assistant for Pedalwise Bikes (the made-up shop
from Tutorial 06). It can search the shop's documents and do arithmetic.

What you'll learn:
  1. Define the graph's state with MessagesState
  2. Add nodes: one that calls the model, one that runs tools (ToolNode)
  3. Connect them with edges, including a conditional "tools or finish?" edge
  4. Add memory with a checkpointer and a thread_id
  5. Persist memory to SQLite so it survives restarts (--chat)

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 07_langgraph_agent.py          # demos
  uv run 07_langgraph_agent.py --chat   # chat; quit and rerun, it remembers you
  uv run 07_langgraph_agent.py --chat --new   # start a fresh conversation
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

HERE = Path(__file__).parent
DB_PATH = HERE / "memory.sqlite"

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Tools (same ideas as Tutorials 05 and 06)
# ---------------------------------------------------------------------------
_store: InMemoryVectorStore | None = None


def get_store() -> InMemoryVectorStore:
    """Build the document index from Tutorial 06 the first time it's needed."""
    global _store
    if _store is None:
        docs = [Document(p.read_text(encoding="utf-8"), metadata={"source": p.name})
                for p in sorted((HERE / "data").glob("*.md"))]
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
        _store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
        _store.add_documents(chunks)
    return _store


@tool
def search_docs(query: str) -> str:
    """Search Pedalwise Bikes' policies (returns, warranty, servicing, prices,
    opening hours). Use for any question about the shop."""
    results = get_store().similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in results)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '120 + 25'. Supports + - * /
    and parentheses. Use this for any maths, e.g. adding up prices."""
    if not set(expression) <= set("0123456789+-*/(). ") or "**" in expression:
        return "Error: only numbers, + - * / and parentheses are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as error:
        return f"Error: {error}"


TOOLS = [search_docs, calculate]
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM = SystemMessage(
    "You are the support assistant for Pedalwise Bikes. Answer shop questions "
    "using search_docs and do maths with calculate. If the documents don't "
    "cover something, say so. Keep answers short and friendly."
)


# ---------------------------------------------------------------------------
# Step 1: State
# ---------------------------------------------------------------------------
# The state is the data that flows through the graph. MessagesState is a
# ready-made state with one key, "messages". When a node returns
# {"messages": [new_message]}, LangGraph *appends* it rather than replacing
# the list, so the conversation accumulates automatically.


# ---------------------------------------------------------------------------
# Step 2: Nodes
# ---------------------------------------------------------------------------
# A node is a function that takes the state and returns an update to it.
def agent(state: MessagesState) -> dict:
    """Call the model with the conversation so far."""
    response = llm_with_tools.invoke([SYSTEM, *state["messages"]])
    return {"messages": [response]}


# ToolNode is a prebuilt node that runs every tool call in the last AI message
# and appends the ToolMessages, exactly what our for-loop did in Tutorial 05.
# It also catches tool exceptions and reports them back to the model.
tool_node = ToolNode(TOOLS)


# ---------------------------------------------------------------------------
# Step 3: Edges
# ---------------------------------------------------------------------------
#   START -> agent -> (asked for tools?) --yes--> tools -> agent -> ...
#                                         --no---> END
# tools_condition is a prebuilt router: it returns "tools" if the last message
# has tool_calls, otherwise END.
def build_graph() -> StateGraph:
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph


def show_graph() -> None:
    print("=== Steps 1-3: the graph (Mermaid syntax, paste into mermaid.live) ===")
    print(build_graph().compile().get_graph().draw_mermaid())


# ---------------------------------------------------------------------------
# Running the graph and watching each step
# ---------------------------------------------------------------------------
def run(app, text: str, thread_id: str) -> None:
    """Send one user message and print each node's output as it happens."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"you: {text}")
    # stream_mode="updates" yields {node_name: state_update} after each node.
    for update in app.stream({"messages": [("user", text)]}, config, stream_mode="updates"):
        for node, change in update.items():
            if node == "tools":  # one ToolMessage per tool call
                for result in change["messages"]:
                    print(f"  [tool {result.name}] {result.content[:70].strip()!r}...")
                continue
            message = change["messages"][-1]
            if message.tool_calls:
                calls = ", ".join(f"{c['name']}({c['args']})" for c in message.tool_calls)
                print(f"  [agent wants] {calls}")
            else:
                print(f"ai : {message.text}")
    print()


# ---------------------------------------------------------------------------
# Step 4: Memory with a checkpointer
# ---------------------------------------------------------------------------
# A checkpointer saves the state after every step, keyed by thread_id. Call
# the graph again with the same thread_id and it continues that conversation;
# a different thread_id is a separate conversation. No history list needed.
def memory_demo() -> None:
    print("=== Step 4: memory with InMemorySaver ===")
    app = build_graph().compile(checkpointer=InMemorySaver())

    run(app, "Hi, I'm Sam. I just bought an e-bike. How long is the battery covered?", "sam")
    run(app, "What would an e-bike service plus express surcharge cost me?", "sam")
    run(app, "What's my name, and what did I buy?", "sam")      # remembered
    run(app, "What's my name?", "someone-else")                 # new thread: unknown

    # You can inspect everything the checkpointer saved for a thread.
    saved = app.get_state({"configurable": {"thread_id": "sam"}})
    print(f"(thread 'sam' has {len(saved.values['messages'])} saved messages)\n")


# ---------------------------------------------------------------------------
# Step 5: Persistent memory with SQLite
# ---------------------------------------------------------------------------
# Swap InMemorySaver for SqliteSaver and the conversation is written to
# memory.sqlite. Quit, rerun with --chat, and the agent still remembers.
# (For production there are Postgres and other checkpointers.)
def chat(new: bool) -> None:
    if new and DB_PATH.exists():
        DB_PATH.unlink()
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = build_graph().compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "cli"}}
        count = len(app.get_state(config).values.get("messages", []))
        print(f"Chat started ({count} messages remembered from earlier runs). "
              f"Type 'quit' to exit.\n")
        while True:
            try:
                text = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                result = app.invoke({"messages": [("user", text)]}, config)
                print("ai :", result["messages"][-1].text, "\n")


if __name__ == "__main__":
    if "--chat" in sys.argv:
        chat(new="--new" in sys.argv)
    else:
        show_graph()
        memory_demo()
