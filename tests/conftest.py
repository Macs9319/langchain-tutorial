"""Shared test setup: a scripted model and offline embeddings.

Tests must be fast, free and repeatable, so they never call OpenAI. The fake
model below follows simple rules based on the last message, which is enough
to drive the real agent, tools, middleware and server end to end.
"""

import hashlib
import json
import os
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pedalwise.config import Settings
from pedalwise.server import create_app


class WordEmbeddings(Embeddings):
    """Hashes words into a vector: crude, but offline and deterministic."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 256
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        return vector

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def _call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


class ScriptedModel(BaseChatModel):
    """Picks a tool from keywords in the user's message, then summarises the
    tool result. Handles the refund flow: eligibility check, then request."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if last.type == "human":
            text = last.text.lower()
            order = re.search(r"a\d{4}", text)
            if "refund" in text and order:
                reply = _call("check_refund_eligibility", order_id=order.group().upper())
            elif "orders" in text:
                reply = _call("my_orders")
            elif "stock" in text:
                reply = _call("check_stock", item="Trail 500", store="Porto")
            else:
                reply = _call("search_policies", query=last.text)
        elif isinstance(last, ToolMessage) and last.name == "check_refund_eligibility" \
                and last.text.startswith("Eligible"):
            order_id = json.dumps(re.findall(r"A\d{4}", str(messages)))  # most recent order id
            reply = _call("request_refund", order_id=json.loads(order_id)[-1], reason="customer request")
        elif isinstance(last, ToolMessage):
            reply = AIMessage(f"Here's what I found: {last.text}")
        else:
            reply = AIMessage("Anything else?")
        # Pretend usage, so token counting (Tutorial 22) can be tested.
        reply.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


STAFF_TOKEN = "test-staff-token"


@pytest.fixture
def db_url(tmp_path):
    """A fresh, empty database for one test.

    SQLite by default. Set TEST_DATABASE_URL=postgresql://... to run the same
    tests on Postgres (Tutorial 26; CI does both): each test then gets its
    own new database, dropped afterwards.
    """
    base = os.getenv("TEST_DATABASE_URL")
    if not base:
        yield str(tmp_path / "test.sqlite")
        return
    import psycopg
    name = f"pedalwise_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    yield f"{base.rsplit('/', 1)[0]}/{name}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE {name} WITH (FORCE)")


def make_settings(tmp_path, db_url, **overrides) -> Settings:
    # database_url="" so a DATABASE_URL in your shell can't point tests at real data.
    return Settings(index_dir=tmp_path / "index", db_path=db_url, database_url="",
                    staff_token=STAFF_TOKEN, **overrides)


def refunds(client) -> list[dict]:
    """The refund ledger, as staff see it."""
    return client.get("/api/staff/refunds", headers={"X-Staff-Token": STAFF_TOKEN}).json()


@pytest.fixture
def client(tmp_path, db_url):
    settings = make_settings(tmp_path, db_url)
    app = create_app(settings, model=ScriptedModel(), fallback_model=None,
                     embeddings=WordEmbeddings())
    with TestClient(app) as test_client:  # runs the lifespan (index, database)
        yield test_client


def read_events(response) -> list[tuple[str, dict]]:
    """Parse a Server-Sent Events body into (event, data) pairs."""
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields.get("event", "message"), json.loads(fields.get("data", "{}"))))
    return events
