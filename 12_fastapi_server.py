"""
Tutorial 12 — Serving your agent as a web API
=============================================

So far every agent lived inside a terminal script. To use one from a
website, a mobile app or another service, put it behind an HTTP API. This
tutorial wraps a LangGraph agent in FastAPI, with three ways to talk to it:

  POST /chat                one request, one JSON reply
  POST /chat/stream         Server-Sent Events: tokens, tool calls, progress
  GET  /threads/{id}        the saved conversation for a thread
  GET  /                    a small chat page that uses /chat/stream

What you'll learn:
  1. Build the agent once at startup with a FastAPI lifespan
  2. Validate requests and shape responses with Pydantic models
  3. Give each conversation a thread_id so the agent remembers it
  4. Stream Tutorial 11's events to the browser as Server-Sent Events (SSE)
  5. Report errors inside a stream, where HTTP status codes no longer work

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 12_fastapi_server.py
  then open http://127.0.0.1:8001 for the chat page,
  or http://127.0.0.1:8001/docs for FastAPI's interactive API docs.

Try it from another terminal:
  curl -s localhost:8001/chat -H 'content-type: application/json' \\
       -d '{"message": "Is the Trail 500 in stock in Lisbon?"}'
  curl -N localhost:8001/chat/stream -H 'content-type: application/json' \\
       -d '{"message": "Where is order A1003?"}'
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# The agent (Tutorials 07, 08 and 11 combined)
# ---------------------------------------------------------------------------
ORDERS = {
    "A1001": {"item": "Trail 500 mountain bike", "status": "delivered 2 days ago"},
    "A1002": {"item": "Commuter helmet", "status": "delivered last month"},
    "A1003": {"item": "City e-bike", "status": "in transit, arriving Friday"},
}
STOCK = {("city e-bike", "porto"): 2, ("city e-bike", "lisbon"): 0,
         ("trail 500", "porto"): 5, ("trail 500", "lisbon"): 3}


@tool
def get_order(order_id: str) -> str:
    """Look up an order by ID (e.g. 'A1001'): item and delivery status."""
    order = ORDERS.get(order_id.strip().upper())
    return str(order) if order else f"No order with ID {order_id}."


@tool
async def check_stock(item: str, store: str) -> str:
    """Check how many units of an item a store has. Stores: Lisbon, Porto.
    Items: 'City e-bike', 'Trail 500'."""
    write = get_stream_writer()
    write(f"Checking the {store} warehouse...")
    await asyncio.sleep(1)  # pretend this is a slow network call
    count = STOCK.get((item.strip().lower(), store.strip().lower()))
    return f"No data for {item} in {store}." if count is None else f"{count} x {item} in {store}"


TOOLS = [get_order, check_stock]
llm_with_tools = llm.bind_tools(TOOLS)
SYSTEM = SystemMessage("You are Pedalwise Bikes support. Use the tools for orders "
                       "and stock. Answer in one to three friendly sentences.")


async def agent(state: MessagesState) -> dict:
    return {"messages": [await llm_with_tools.ainvoke([SYSTEM, *state["messages"]])]}


def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    # InMemorySaver forgets everything when the server stops. For memory that
    # survives restarts in an async server, use AsyncSqliteSaver (package
    # langgraph-checkpoint-sqlite, needs aiosqlite) or AsyncPostgresSaver.
    return graph.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Step 1: Build the agent once, at startup
# ---------------------------------------------------------------------------
# The lifespan function runs once when the server starts (before `yield`)
# and once when it stops (after). Building the graph here, not inside each
# request, means every request shares one compiled graph and one memory.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    yield


app = FastAPI(title="Pedalwise support agent", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Step 2: Request and response models
# ---------------------------------------------------------------------------
# FastAPI checks every request against these models and answers 422 with a
# clear error if it doesn't match: no hand-written validation needed. They
# also appear in the interactive docs at /docs.
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="The user's message")
    thread_id: str | None = Field(
        default=None, description="Continue this conversation; omit to start a new one")


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


class HistoryMessage(BaseModel):
    role: str
    text: str


# ---------------------------------------------------------------------------
# Step 3: A JSON endpoint with per-conversation memory
# ---------------------------------------------------------------------------
# The thread_id is the conversation's key in the checkpointer (Tutorial 07).
# The client stores it and sends it back to continue the same conversation.
def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    result = await app.state.graph.ainvoke(
        {"messages": [("user", request.message)]}, config_for(thread_id))
    return ChatResponse(thread_id=thread_id, reply=result["messages"][-1].text)


@app.get("/threads/{thread_id}", response_model=list[HistoryMessage])
async def history(thread_id: str) -> list[HistoryMessage]:
    snapshot = await app.state.graph.aget_state(config_for(thread_id))
    messages = snapshot.values.get("messages", [])
    if not messages:
        raise HTTPException(status_code=404, detail="Unknown thread")
    # Show only what a chat window would: user messages and final AI replies.
    return [HistoryMessage(role=m.type, text=m.text) for m in messages
            if m.type == "human" or (m.type == "ai" and not m.tool_calls)]


# ---------------------------------------------------------------------------
# Step 4: Streaming with Server-Sent Events
# ---------------------------------------------------------------------------
# SSE is a simple text format over a normal HTTP response that stays open:
#     event: token
#     data: {"text": "Hel"}
#     (blank line ends each event)
# With response_class=EventSourceResponse, FastAPI turns every
# ServerSentEvent we `yield` into that format and sends it immediately.
async def agent_events(graph, message: str, thread_id: str) -> AsyncIterator[dict]:
    """Tutorial 11's simplified events, now for a given thread."""
    inputs = {"messages": [("user", message)]}
    async for mode, data in graph.astream(inputs, config_for(thread_id),
                                          stream_mode=["messages", "updates", "custom"]):
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


@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(request: ChatRequest) -> AsyncIterator[ServerSentEvent]:
    thread_id = request.thread_id or str(uuid.uuid4())
    # First event: tell the client which thread this is, so it can continue it.
    yield ServerSentEvent(event="thread", data={"thread_id": thread_id})
    try:
        async with asyncio.timeout(120):  # never let one request hang forever
            async for item in agent_events(app.state.graph, request.message, thread_id):
                kind = item.pop("type")
                yield ServerSentEvent(event=kind, data=item)
    # -----------------------------------------------------------------------
    # Step 5: Errors inside a stream
    # -----------------------------------------------------------------------
    # The 200 status was sent with the first event, so an exception can't
    # become a 500 any more. Send an "error" event the client can display.
    # Log the details on the server; don't send internals to users.
    except TimeoutError:
        yield ServerSentEvent(event="error", data={"message": "The agent took too long."})
    except Exception as error:
        print(f"[stream error] thread={thread_id}: {error!r}")
        yield ServerSentEvent(event="error", data={"message": "Something went wrong."})
    yield ServerSentEvent(event="done", data={})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# A minimal chat page that reads /chat/stream
# ---------------------------------------------------------------------------
# The browser's EventSource only supports GET, so the page uses fetch() and
# parses the SSE text itself: split on blank lines, read "event:" and "data:".
CHAT_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pedalwise Support</title>
<style>
  :root { --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --me:#dbe8ff; --bot:#fff; --line:#e2e2dc; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --fg:#ededea; --muted:#9a9a93; --me:#23344f; --bot:#222220; --line:#33332f; } }
  body { margin:0; font:16px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  main { max-width:680px; margin:0 auto; padding:16px; display:flex; flex-direction:column; height:100dvh; box-sizing:border-box; }
  h1 { font-size:18px; margin:0 0 12px; }
  #log { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
  .msg { padding:8px 12px; border-radius:12px; max-width:85%; white-space:pre-wrap; border:1px solid var(--line); }
  .me { background:var(--me); align-self:flex-end; }
  .bot { background:var(--bot); align-self:flex-start; }
  .step { color:var(--muted); font-size:13px; align-self:flex-start; }
  form { display:flex; gap:8px; margin-top:12px; }
  input { flex:1; font:inherit; padding:10px 12px; border-radius:10px; border:1px solid var(--line); background:var(--bot); color:var(--fg); }
  button { font:inherit; padding:10px 16px; border-radius:10px; border:0; background:#2f6fde; color:#fff; }
  button:disabled { opacity:.5; }
</style></head>
<body><main>
  <h1>Pedalwise Bikes support</h1>
  <div id="log"><div class="step">Try: "Is the City e-bike in stock in Porto?" or "Where is order A1003?"</div></div>
  <form id="form"><input id="text" autocomplete="off" placeholder="Ask about orders or stock" required>
  <button id="send">Send</button></form>
</main>
<script>
let threadId = null;
const log = document.getElementById("log");
function add(cls, text) {
  const div = document.createElement("div");
  div.className = cls; div.textContent = text;   // textContent: never inject HTML
  log.appendChild(div); log.scrollTop = log.scrollHeight; return div;
}
document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("text"), button = document.getElementById("send");
  const message = input.value.trim(); if (!message) return;
  input.value = ""; button.disabled = true; add("msg me", message);
  let reply = null;
  try {
    const response = await fetch("/chat/stream", { method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }) });
    if (!response.ok) { add("step", "Request failed: " + response.status); return; }
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buffer += value;
      const blocks = buffer.split("\\n\\n"); buffer = blocks.pop();  // keep partial event
      for (const block of blocks) {
        let event = "message", data = "";
        for (const line of block.split("\\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        const payload = data ? JSON.parse(data) : {};
        if (event === "thread") threadId = payload.thread_id;
        else if (event === "token") { reply = reply || add("msg bot", ""); reply.textContent += payload.text; }
        else if (event === "tool_call") add("step", "→ " + payload.name + " " + JSON.stringify(payload.args));
        else if (event === "progress") add("step", "… " + payload.text);
        else if (event === "tool_result") { add("step", "← " + payload.text); reply = null; }
        else if (event === "error") add("step", "⚠ " + payload.message);
      }
    }
  } catch (err) { add("step", "Connection error: " + err.message); }
  finally { button.disabled = false; input.focus(); }
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat_page() -> str:
    return CHAT_PAGE


if __name__ == "__main__":
    # 127.0.0.1 means only this computer can connect. Use host="0.0.0.0" to
    # allow other devices on your network, but add authentication first:
    # anyone who can reach the server can spend your OpenAI credits.
    uvicorn.run(app, host="127.0.0.1", port=8001)
