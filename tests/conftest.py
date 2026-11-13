"""Shared test setup: a scripted model and offline embeddings.

Tests must be fast, free and repeatable, so they never call OpenAI. The fake
model below follows simple rules based on the last message, which is enough
to drive the real agent, tools, middleware and server end to end.
"""

import hashlib
import json
import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pedalwise import shop
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
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


STAFF_TOKEN = "test-staff-token"


@pytest.fixture
def client(tmp_path):
    shop.REFUNDS.clear()
    settings = Settings(index_dir=tmp_path / "index", db_path=str(tmp_path / "test.sqlite"),
                        staff_token=STAFF_TOKEN)
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
