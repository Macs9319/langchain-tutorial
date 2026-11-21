"""The web app: customer chat (streamed), saved conversations, staff approvals."""

import asyncio
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from .agent import build_agent
from .approvals import ApprovalQueue
from .config import Settings
from .knowledge import KnowledgeBase
from .shop import DEMO_CUSTOMERS, Customer
from .tools import make_tools

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger("pedalwise")
FRIENDLY = {"search_policies": "Looking that up…", "my_orders": "Checking your orders…",
            "check_stock": "Checking stock…", "check_refund_eligibility": "Checking the order…",
            "request_refund": "Requesting a refund…"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    customer_email: str
    thread_id: str | None = None


class Decision(BaseModel):
    approve: bool
    note: str | None = Field(default=None, max_length=500)


def create_app(settings: Settings | None = None, model: BaseChatModel | None = None,
               fallback_model: BaseChatModel | None = None,
               embeddings: Embeddings | None = None) -> FastAPI:
    """Build the app. Tests pass fake models and embeddings; normally the
    OpenAI ones are created from settings."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Building the index may call the embeddings API: run it off the
        # event loop so the server stays responsive.
        kb = await asyncio.to_thread(
            KnowledgeBase, settings.data_dir, settings.index_dir,
            embeddings or OpenAIEmbeddings(model="text-embedding-3-small"))
        primary = model or ChatOpenAI(model=settings.model, max_tokens=4096, timeout=60)
        fallback = fallback_model if model else ChatOpenAI(
            model=settings.fallback_model, max_tokens=4096, timeout=60)
        # Conversations are saved in SQLite, so they survive restarts (07).
        async with AsyncSqliteSaver.from_conn_string(settings.db_path) as checkpointer:
            app.state.agent = build_agent(primary, fallback, make_tools(kb), checkpointer)
            # Conversations waiting for staff, also in SQLite (21).
            app.state.pending = await ApprovalQueue.open(settings.db_path)
            if settings.staff_token_generated:
                logger.warning("No STAFF_TOKEN set. Staff token for this run: %s",
                               settings.staff_token)
            logger.info("Ready", extra={"environment": settings.environment,
                                        "model": settings.model})
            yield
            await app.state.pending.close()
            logger.info("Shut down cleanly")

    # The interactive API docs help while developing; in production they just
    # advertise every endpoint to strangers.
    docs = ({"docs_url": None, "redoc_url": None, "openapi_url": None}
            if settings.production else {})
    app = FastAPI(title="Pedalwise support", lifespan=lifespan, **docs)

    # -- dependencies: checks that run before an endpoint does --------------
    def current_customer(email: str) -> Customer:
        # A real app would read this from a login session, not trust the client.
        customer = DEMO_CUSTOMERS.get(email.strip().lower())
        if customer is None:
            raise HTTPException(401, "Unknown customer")
        return customer

    def require_staff(x_staff_token: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_staff_token, settings.staff_token):
            raise HTTPException(401, "Staff token required")

    def config_for(thread_id: str, customer: Customer) -> dict:
        # The owner travels with every call, so it's saved with every checkpoint.
        return {"configurable": {"thread_id": thread_id},
                "metadata": {"customer": customer.email}}

    async def owner_of(thread_id: str) -> str | None:
        snapshot = await app.state.agent.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.metadata.get("customer") if snapshot.metadata else None

    async def chat_request(request: ChatRequest) -> tuple[ChatRequest, Customer, str]:
        customer = current_customer(request.customer_email)
        if request.thread_id:
            # Someone else's conversation looks exactly like a missing one.
            if await owner_of(request.thread_id) != customer.email:
                raise HTTPException(404, "Unknown conversation")
            if await app.state.pending.has(request.thread_id):
                raise HTTPException(409, "Waiting for a staff member to review a refund")
        return request, customer, request.thread_id or str(uuid.uuid4())

    # -- health checks (Tutorial 21) -------------------------------------------
    # Platforms poll these. /healthz: "the process is alive" (restart it if
    # not). /readyz: "it can serve requests" (send it traffic only if so).
    # Neither calls OpenAI: a probe every few seconds would cost money, and an
    # OpenAI outage shouldn't make the platform restart a healthy server.
    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> dict:
        try:
            await app.state.pending.ping()
        except Exception as error:
            logger.error("Readiness check failed: %r", error)
            raise HTTPException(503, "Database unavailable")
        return {"status": "ready"}

    # -- pages ----------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC / "chat.html")

    @app.get("/staff", include_in_schema=False)
    async def staff_page() -> FileResponse:
        return FileResponse(STATIC / "staff.html")

    @app.get("/api/customers")
    async def customers() -> list[dict]:
        return [{"email": c.email, "name": c.name} for c in DEMO_CUSTOMERS.values()]

    # -- customer chat (Tutorials 11 and 12) ----------------------------------
    @app.post("/api/chat", response_class=EventSourceResponse)
    async def chat(prepared=Depends(chat_request)) -> AsyncIterator[ServerSentEvent]:
        request, customer, thread_id = prepared
        yield ServerSentEvent(event="thread", data={"thread_id": thread_id})
        inputs = {"messages": [{"role": "user", "content": request.message}]}
        try:
            async with asyncio.timeout(120):
                async for mode, data in app.state.agent.astream(
                        inputs, config_for(thread_id, customer), context=customer,
                        stream_mode=["messages", "updates", "custom"]):
                    if mode == "messages":
                        chunk, meta = data
                        # Streaming models send pieces (AIMessageChunk); others send the
                        # whole reply at once (AIMessage). The check accepts both.
                        if (meta.get("langgraph_node") == "model"
                                and isinstance(chunk, AIMessage) and chunk.text):
                            yield ServerSentEvent(event="token", data={"text": chunk.text})
                    elif mode == "custom":
                        yield ServerSentEvent(event="status", data={"text": str(data)})
                    elif mode == "updates" and "__interrupt__" in data:
                        actions = data["__interrupt__"][0].value["action_requests"]
                        await app.state.pending.add(thread_id, customer.email, actions)
                        logger.info("Refund waiting for staff", extra={"thread_id": thread_id})
                        yield ServerSentEvent(event="approval_pending", data={"actions": actions})
                    elif mode == "updates":
                        for update in data.values():
                            for message in (update or {}).get("messages", []):
                                for call in getattr(message, "tool_calls", []):
                                    text = FRIENDLY.get(call["name"], "Working…")
                                    yield ServerSentEvent(event="status", data={"text": text})
        except TimeoutError:
            yield ServerSentEvent(event="error", data={"message": "That took too long."})
        except Exception:  # logged here, never shown to the customer
            logger.exception("Chat failed", extra={"thread_id": thread_id})
            yield ServerSentEvent(event="error", data={"message": "Something went wrong."})
        yield ServerSentEvent(event="done", data={})

    @app.get("/api/threads/{thread_id}")
    async def thread(thread_id: str, customer_email: str) -> dict:
        customer = current_customer(customer_email)
        if await owner_of(thread_id) != customer.email:
            raise HTTPException(404, "Unknown conversation")
        snapshot = await app.state.agent.aget_state({"configurable": {"thread_id": thread_id}})
        messages = [{"role": m.type, "text": m.text} for m in snapshot.values["messages"]
                    if m.type == "human" or (m.type == "ai" and m.text and not m.tool_calls)]
        return {"messages": messages, "pending": await app.state.pending.has(thread_id)}

    # -- staff approvals (Tutorials 08 and 13) ---------------------------------
    @app.get("/api/staff/approvals", dependencies=[Depends(require_staff)])
    async def approvals() -> list[dict]:
        return await app.state.pending.all()

    @app.post("/api/staff/approvals/{thread_id}", dependencies=[Depends(require_staff)])
    async def decide(thread_id: str, decision: Decision) -> dict:
        # Take it off the queue first so a double-click can't decide twice;
        # put it back on failure so the refund doesn't silently drop out.
        item = await app.state.pending.take(thread_id)
        if item is None:
            raise HTTPException(404, "Nothing waiting for approval here")
        customer = DEMO_CUSTOMERS[item["customer"]]
        verdict = ({"type": "approve"} if decision.approve else
                   {"type": "reject", "message": decision.note or "Declined by staff."})
        # An audit trail: who got what, and when, ends up in the logs.
        logger.info("Refund decided", extra={"thread_id": thread_id, "approved": decision.approve,
                                             "customer": customer.email})
        try:
            result = await app.state.agent.ainvoke(
                Command(resume={"decisions": [verdict] * len(item["actions"])}),
                config_for(thread_id, customer), context=customer)
        except Exception:
            await app.state.pending.add(thread_id, item["customer"], item["actions"])
            raise
        if "__interrupt__" in result:  # the agent asked for another approval
            await app.state.pending.add(thread_id, customer.email,
                                        result["__interrupt__"][0].value["action_requests"])
        return {"reply": result["messages"][-1].text}

    return app
