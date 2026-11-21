"""The web app: customer chat (streamed), saved conversations, staff approvals."""

import asyncio
import logging
import secrets
import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langsmith import Client as LangSmithClient
from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel, Field

from .agent import build_agent
from .approvals import ApprovalQueue
from .config import Settings
from .database import Database, is_postgres_url
from .knowledge import KnowledgeBase
from .models import chat_model
from .refunds import RefundLedger
from .shop import DEMO_CUSTOMERS, Customer
from .telemetry import RunRecorder, Telemetry
from .tools import make_tools

STATIC = Path(__file__).parent / "static"
# Which copy of the app this is: the container's hostname (Tutorial 26).
REPLICA = socket.gethostname()
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


class Feedback(BaseModel):
    run_id: uuid.UUID
    customer_email: str
    score: Literal[1, -1]  # thumbs up or down
    comment: str | None = Field(default=None, max_length=1000)


@asynccontextmanager
async def open_checkpointer(url: str, db: Database) -> AsyncIterator:
    """Saved conversations: SQLite for one copy, Postgres for several (07, 26)."""
    if not is_postgres_url(url):
        async with AsyncSqliteSaver.from_conn_string(url) as checkpointer:
            yield checkpointer
        return
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    # A pool, so conversations on this copy don't queue for one connection.
    # These connection settings are what the Postgres checkpointer expects.
    async with AsyncConnectionPool(url, max_size=10, open=False, kwargs={
            "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        async with db.schema_lock():  # one copy at a time creates the tables
            await checkpointer.setup()
        yield checkpointer


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
        if embeddings is None and settings.embeddings == "openai":
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            embeddings_model = embeddings  # None: keyword search only (23)
        kb = await asyncio.to_thread(
            KnowledgeBase, settings.data_dir, settings.index_dir, embeddings_model)
        # Any provider, from a "provider:model" string (23).
        def make(spec: str) -> BaseChatModel:
            return chat_model(spec, deepseek_thinking=settings.deepseek_thinking,
                              max_tokens=4096, timeout=60)
        primary = model or make(settings.model)
        if model:  # tests pass their own models
            fallback = fallback_model
        else:
            fallback = None if settings.fallback_model == "none" else make(settings.fallback_model)
        # Everything the app writes lives in one database, shared by every
        # copy of the app: SQLite locally, Postgres when scaled out (26).
        db = await Database.connect(settings.db_url)
        app.state.pending = ApprovalQueue(db)      # refunds waiting for staff (21)
        app.state.ledger = RefundLedger(db)        # refunds paid, once each (26)
        app.state.telemetry = Telemetry(db)        # runs and feedback (22)
        async with db.schema_lock():
            for store in (app.state.pending, app.state.ledger, app.state.telemetry):
                await store.setup()
        async with open_checkpointer(settings.db_url, db) as checkpointer:
            app.state.agent = build_agent(primary, fallback,
                                          make_tools(kb, app.state.ledger), checkpointer)
            purged = await app.state.telemetry.delete_older_than(settings.telemetry_retention_days)
            if purged:
                logger.info("Deleted old runs", extra={"runs": purged})
            if settings.staff_token_generated:
                logger.warning("No STAFF_TOKEN set. Staff token for this run: %s",
                               settings.staff_token)
            logger.info("Ready", extra={"environment": settings.environment,
                                        "model": settings.model, "replica": REPLICA,
                                        "database": "postgres" if db.postgres else "sqlite"})
            yield
        await db.close()
        logger.info("Shut down cleanly")

    # The interactive API docs help while developing; in production they just
    # advertise every endpoint to strangers.
    docs = ({"docs_url": None, "redoc_url": None, "openapi_url": None}
            if settings.production else {})
    app = FastAPI(title="Pedalwise support", lifespan=lifespan, **docs)

    @app.middleware("http")
    async def served_by(request: Request, call_next):
        # Label every response with the copy that served it, so you can see
        # a load balancer spreading requests (and find the right logs).
        response = await call_next(request)
        response.headers["X-Served-By"] = REPLICA
        return response

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

    def traced(config: dict, run_id: uuid.UUID, recorder: RunRecorder, name: str) -> dict:
        """Add what observability needs to a run's config (22). The run_id
        we choose becomes the trace's id in LangSmith, so feedback sent later
        lands on the right trace. Tags and metadata make traces filterable."""
        return {**config, "run_id": run_id, "run_name": name, "callbacks": [recorder],
                "tags": [settings.environment, name]}

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
        run_id, recorder = uuid.uuid4(), RunRecorder()
        answer, outcome = [], "ok"
        yield ServerSentEvent(event="thread", data={"thread_id": thread_id})
        # The browser needs the run id to send feedback about this answer.
        yield ServerSentEvent(event="run", data={"run_id": str(run_id)})
        inputs = {"messages": [{"role": "user", "content": request.message}]}
        config = traced(config_for(thread_id, customer), run_id, recorder, "support-chat")
        try:
            async with asyncio.timeout(120):
                async for mode, data in app.state.agent.astream(
                        inputs, config, context=customer,
                        stream_mode=["messages", "updates", "custom"]):
                    if mode == "messages":
                        chunk, meta = data
                        # Streaming models send pieces (AIMessageChunk); others send the
                        # whole reply at once (AIMessage). The check accepts both.
                        if (meta.get("langgraph_node") == "model"
                                and isinstance(chunk, AIMessage) and chunk.text):
                            answer.append(chunk.text)
                            yield ServerSentEvent(event="token", data={"text": chunk.text})
                    elif mode == "custom":
                        yield ServerSentEvent(event="status", data={"text": str(data)})
                    elif mode == "updates" and "__interrupt__" in data:
                        actions = data["__interrupt__"][0].value["action_requests"]
                        await app.state.pending.add(thread_id, customer.email, actions)
                        outcome = "approval"
                        logger.info("Refund waiting for staff", extra={"thread_id": thread_id})
                        yield ServerSentEvent(event="approval_pending", data={"actions": actions})
                    elif mode == "updates":
                        for update in data.values():
                            for message in (update or {}).get("messages", []):
                                for call in getattr(message, "tool_calls", []):
                                    text = FRIENDLY.get(call["name"], "Working…")
                                    yield ServerSentEvent(event="status", data={"text": text})
        except TimeoutError:
            outcome = "timeout"
            yield ServerSentEvent(event="error", data={"message": "That took too long."})
        except Exception:  # logged here, never shown to the customer
            outcome = "error"
            logger.exception("Chat failed", extra={"thread_id": thread_id, "run_id": str(run_id)})
            yield ServerSentEvent(event="error", data={"message": "Something went wrong."})
        # Recorded before "done". If the customer closes the page mid-answer,
        # the stream is cancelled and the run isn't recorded: a known gap.
        await app.state.telemetry.record(
            str(run_id), thread_id, customer.email, "chat", outcome, recorder,
            request.message, "".join(answer))
        yield ServerSentEvent(event="done", data={})

    # -- feedback (Tutorial 22) ------------------------------------------------
    @app.post("/api/feedback")
    async def feedback(item: Feedback) -> dict:
        customer = current_customer(item.customer_email)
        run_id = str(item.run_id)
        if await app.state.telemetry.owner(run_id) != customer.email:
            raise HTTPException(404, "Unknown run")
        await app.state.telemetry.add_feedback(run_id, item.score, item.comment)
        logger.info("Feedback", extra={"run_id": run_id, "score": item.score})
        if tracing_is_enabled():
            # Attach it to the trace in LangSmith as well. The SDK call is
            # blocking, so run it in a thread; a LangSmith outage must never
            # break the customer's click, so failures are only logged.
            try:
                await asyncio.to_thread(LangSmithClient().create_feedback, run_id,
                                        key="user_score", score=item.score, comment=item.comment)
            except Exception as error:
                logger.warning("Couldn't send feedback to LangSmith: %r", error)
        return {"ok": True}

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

    @app.get("/api/staff/refunds", dependencies=[Depends(require_staff)])
    async def refunds() -> list[dict]:
        return await app.state.ledger.all()

    @app.get("/api/staff/metrics", dependencies=[Depends(require_staff)])
    async def metrics(hours: int = 24) -> dict:
        report = await app.state.telemetry.metrics(max(1, min(hours, 24 * 90)))
        report["negative_feedback"] = await app.state.telemetry.negative_feedback(limit=10)
        return report

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
        run_id, recorder = uuid.uuid4(), RunRecorder()
        config = traced(config_for(thread_id, customer), run_id, recorder, "staff-decision")
        try:
            result = await app.state.agent.ainvoke(
                Command(resume={"decisions": [verdict] * len(item["actions"])}),
                config, context=customer)
        except Exception:
            await app.state.pending.add(thread_id, item["customer"], item["actions"])
            await app.state.telemetry.record(str(run_id), thread_id, customer.email,
                                             "staff_decision", "error", recorder, None, None)
            raise
        await app.state.telemetry.record(
            str(run_id), thread_id, customer.email, "staff_decision",
            "approval" if "__interrupt__" in result else "ok", recorder,
            None, result["messages"][-1].text)
        if "__interrupt__" in result:  # the agent asked for another approval
            await app.state.pending.add(thread_id, customer.email,
                                        result["__interrupt__"][0].value["action_requests"])
        return {"reply": result["messages"][-1].text}

    return app
