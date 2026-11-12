"""
Tutorial 15 — Long-term memory
==============================

The memory in Tutorials 04 and 07 belonged to one conversation (thread).
Start a new thread and the agent knows nothing about you. Long-term memory
is different: facts about a user that the agent keeps *across* all their
conversations, like "rides gravel" or "prefers email over phone".

LangGraph keeps the two apart:
  - checkpointer -> short-term: the messages of one thread
  - store        -> long-term: facts saved under a namespace, e.g. per user,
                    searchable by meaning (embeddings, as in Tutorial 06)

What you'll learn:
  1. Save, read and search the store directly
  2. Give an agent tools that save and recall memories for the current user
  3. Recall facts in a brand-new conversation, and keep users separate
  4. Inject relevant memories into the prompt automatically, with no tool call
  5. Persist memories to SQLite and let users see and delete them (--chat)

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 15_long_term_memory.py                    # demos (in-memory store)
  uv run 15_long_term_memory.py --chat --user sam  # chat; memories saved to disk
  uv run 15_long_term_memory.py --show --user sam  # list sam's saved memories
"""

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

load_dotenv()

HERE = Path(__file__).parent
DB_PATH = HERE / "long_term_memory.sqlite"

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)

# The index makes the store searchable by meaning: each saved memory's
# "text" field is embedded, and search(query=...) finds the closest ones.
# text-embedding-3-small produces vectors with 1536 numbers ("dims").
INDEX = {
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "dims": 1536,
    "fields": ["text"],
}


def namespace(user_id: str) -> tuple[str, ...]:
    """Memories are filed under a namespace, like a folder path. One per user
    means one user's facts can never show up in another user's search."""
    return ("memories", user_id)


# ---------------------------------------------------------------------------
# Step 1: The store on its own
# ---------------------------------------------------------------------------
def store_basics() -> None:
    print("=== Step 1: store basics ===")
    store = InMemoryStore(index=INDEX)
    ns = namespace("demo-user")

    # put(namespace, key, value): value is any JSON-like dict.
    store.put(ns, "bike", {"text": "Owns a Trail 500 mountain bike"})
    store.put(ns, "contact", {"text": "Prefers to be contacted by email"})
    store.put(ns, "riding", {"text": "Rides gravel trails most weekends"})

    print("get   :", store.get(ns, "bike").value)
    # search() ranks memories by how close their meaning is to the query,
    # even without shared words ("phone call" vs "contacted by email").
    for item in store.search(ns, query="Should I phone them?", limit=2):
        print(f"search: {item.score:.2f}  {item.value['text']}")
    print()


# ---------------------------------------------------------------------------
# Step 2: Memory tools for an agent
# ---------------------------------------------------------------------------
# Who is talking comes from runtime context (Tutorial 13), never from the
# model, so the model can't read or write another user's memories even if
# a message tries to trick it.
@dataclass
class UserContext:
    user_id: str


# A parameter typed ToolRuntime is filled in by LangChain and hidden from the
# model: it only sees `fact` / `query`. runtime.store is the agent's store.
@tool
def save_memory(fact: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a lasting fact about the user (preferences, bikes they own, plans)
    so you remember it in future conversations. One short fact per call."""
    memory_id = uuid.uuid4().hex[:8]
    runtime.store.put(namespace(runtime.context.user_id), memory_id, {"text": fact})
    return f"Saved memory {memory_id}: {fact}"


@tool
def recall_memories(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search what you remember about the user. Use before giving advice that
    could depend on their bikes, preferences or plans."""
    items = runtime.store.search(namespace(runtime.context.user_id), query=query, limit=5)
    if not items:
        return "No memories found."
    return "\n".join(f"[{i.key}] {i.value['text']}" for i in items)


@tool
def forget_memory(memory_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Delete a memory by its ID when the user asks you to forget something
    or a fact is no longer true. Find IDs with recall_memories."""
    ns = namespace(runtime.context.user_id)
    if runtime.store.get(ns, memory_id) is None:
        return f"No memory with ID {memory_id}."
    runtime.store.delete(ns, memory_id)
    return f"Forgot memory {memory_id}."


SYSTEM_PROMPT = (
    "You are Pedalwise Bikes support. You have long-term memory tools. When the "
    "user shares a lasting fact about themselves, save it. When advice could "
    "depend on what you know about them, recall first. Keep answers short."
)


def build_agent(store: BaseStore, middleware=()):
    return create_agent(
        llm,
        tools=[save_memory, recall_memories, forget_memory],
        system_prompt=SYSTEM_PROMPT,
        middleware=list(middleware),
        context_schema=UserContext,
        checkpointer=InMemorySaver(),  # short-term: per thread
        store=store,                   # long-term: across threads
    )


def ask(agent, user_id: str, thread_id: str, text: str) -> None:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread_id}},
        context=UserContext(user_id=user_id),
    )
    # Tools used in this turn only: everything after the last user message.
    last_human = max(i for i, m in enumerate(result["messages"]) if m.type == "human")
    used = [c["name"] for m in result["messages"][last_human:] for c in getattr(m, "tool_calls", [])]
    print(f"{user_id}: {text}")
    print(f"agent: {result['messages'][-1].text}"
          + (f"   (tools: {', '.join(used)})" if used else ""))


# ---------------------------------------------------------------------------
# Step 3: Remembering across conversations
# ---------------------------------------------------------------------------
def across_threads_demo() -> None:
    print("=== Step 3: memory across conversations ===")
    store = InMemoryStore(index=INDEX)
    agent = build_agent(store)

    print("-- Monday, conversation 1 --")
    ask(agent, "sam", "sam-monday",
        "Hi! I ride a City e-bike to work every day, and I'm planning a "
        "trip to the Alps in June.")

    print("-- Friday, a brand-new conversation (no shared messages) --")
    ask(agent, "sam", "sam-friday", "Which service package would you recommend for me?")

    print("-- A different user asks the same thing --")
    ask(agent, "alex", "alex-friday", "Which service package would you recommend for me?")

    print("\nstored for sam :", [i.value["text"] for i in store.search(namespace("sam"))])
    print("stored for alex:", [i.value["text"] for i in store.search(namespace("alex"))], "\n")


# ---------------------------------------------------------------------------
# Step 4: Automatic recall with middleware
# ---------------------------------------------------------------------------
# Relying on the model to call recall_memories means it sometimes won't.
# This middleware searches the store before every model call and puts the
# best matches straight into the system prompt.
@dynamic_prompt
def prompt_with_memories(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    last_user_text = next((m.text for m in reversed(request.state["messages"])
                           if m.type == "human"), "")
    items = request.runtime.store.search(namespace(user_id), query=last_user_text, limit=3)
    if not items:
        return SYSTEM_PROMPT
    remembered = "\n".join(f"- [{i.key}] {i.value['text']}" for i in items)
    return f"{SYSTEM_PROMPT}\n\nWhat you remember about this user:\n{remembered}"


def auto_recall_demo() -> None:
    print("=== Step 4: memories injected automatically ===")
    store = InMemoryStore(index=INDEX)
    facts = ["Owns a Trail 500 mountain bike", "Had the brake pads replaced in March",
             "Is a gold member", "Favourite colour is green", "Allergic to bees",
             "Lives near the Porto store"]
    for n, fact in enumerate(facts, start=1):
        store.put(namespace("jordan"), f"m{n}", {"text": fact})
    # Only the 3 memories closest in meaning to the question reach the prompt;
    # the colour and the bees should stay out.
    agent = build_agent(store, middleware=[prompt_with_memories])
    ask(agent, "jordan", "jordan-1", "My brakes feel soft. What should I do?")
    print()


# ---------------------------------------------------------------------------
# Step 5: Persistent memory, and the user's right to see and delete it
# ---------------------------------------------------------------------------
# SqliteStore keeps memories in long_term_memory.sqlite between runs. Storing
# personal facts comes with responsibilities: tell users what you keep, let
# them see it, and let them delete it (forget_memory, and --show below).
def open_sqlite_store():
    return SqliteStore.from_conn_string(str(DB_PATH), index=INDEX)


def show_memories(user_id: str) -> None:
    with open_sqlite_store() as store:
        store.setup()  # creates the tables on first use
        items = store.search(namespace(user_id), limit=100)
        print(f"{len(items)} {'memory' if len(items) == 1 else 'memories'} saved for {user_id}:")
        for item in items:
            print(f"  [{item.key}] {item.value['text']}")


def chat(user_id: str) -> None:
    with open_sqlite_store() as store:
        store.setup()
        agent = build_agent(store, middleware=[prompt_with_memories])
        thread_id = f"{user_id}-{uuid.uuid4().hex[:6]}"  # new conversation each run
        count = len(store.search(namespace(user_id), limit=100))
        print(f"Chatting as {user_id}. New conversation, {count} long-term memories. "
              f"Type 'quit' to exit.\n")
        while True:
            try:
                text = input(f"{user_id}: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                result = agent.invoke({"messages": [{"role": "user", "content": text}]},
                                      {"configurable": {"thread_id": thread_id}},
                                      context=UserContext(user_id=user_id))
                print("agent:", result["messages"][-1].text, "\n")


def arg_after(flag: str, default: str) -> str:
    if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(flag) + 1]
    return default


if __name__ == "__main__":
    user = arg_after("--user", "sam")
    if "--show" in sys.argv:
        show_memories(user)
    elif "--chat" in sys.argv:
        chat(user)
    else:
        store_basics()
        across_threads_demo()
        auto_recall_demo()
