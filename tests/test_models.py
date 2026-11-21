"""Tests for Tutorial 23: choosing the provider in config, and DeepSeek's quirks.

A fake DeepSeek API (an httpx transport, no network) enforces the rule from
DeepSeek's thinking-mode docs: with tools in the request, every earlier
assistant turn must carry its reasoning_content, or the request fails (400).
The real API didn't enforce it when tested (November 2026); the fake does,
so the code keeps to the documented contract either way.
"""

import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

from pedalwise.config import Settings
from pedalwise.models import ThinkingDeepSeek, chat_model, split_spec
from pedalwise.server import create_app

from .conftest import STAFF_TOKEN, ScriptedModel, read_events


def test_split_spec():
    assert split_spec("deepseek:deepseek-flash") == ("deepseek", "deepseek-flash")
    assert split_spec("gpt-5-mini") == ("openai", "gpt-5-mini")  # old OPENAI_MODEL values


def test_deepseek_thinking_is_off_unless_asked_for():
    def thinking(model):
        return model._get_request_payload([HumanMessage("hi")])["extra_body"]["thinking"]["type"]
    assert thinking(chat_model("deepseek:deepseek-flash", api_key="test")) == "disabled"
    assert thinking(chat_model("deepseek:deepseek-flash", deepseek_thinking=True,
                               api_key="test")) == "enabled"


def test_reasoning_is_sent_back():
    model = chat_model("deepseek:deepseek-flash", api_key="test")
    payload = model._get_request_payload([
        HumanMessage("Stock?"),
        AIMessage("", additional_kwargs={"reasoning_content": "I should check."},
                  tool_calls=[{"name": "check_stock", "args": {}, "id": "c1"}]),
        ToolMessage("5 left", tool_call_id="c1")])
    assert payload["messages"][1]["reasoning_content"] == "I should check."


def fake_deepseek(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("tools"):
        for message in body["messages"]:
            if message["role"] == "assistant" and not message.get("reasoning_content"):
                return httpx.Response(400, json={"error": {
                    "message": "reasoning_content must be passed back", "type": "invalid_request"}})
    if body["messages"][-1]["role"] == "user":
        message = {"role": "assistant", "content": None, "reasoning_content": "Check stock first.",
                   "tool_calls": [{"id": "c1", "type": "function", "function": {
                       "name": "check_stock", "arguments": json.dumps({"item": "Trail 500"})}}]}
    else:
        message = {"role": "assistant", "content": "5 in stock.",
                   "reasoning_content": "The tool said 5."}
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


@tool
def check_stock(item: str) -> str:
    """Check stock of a bike."""
    return f"5 x {item}"


def agent_with(model_class):
    model = model_class(model="deepseek-flash", api_key="test", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(fake_deepseek)))
    return create_agent(model, tools=[check_stock])


def test_agent_with_tools_works_in_thinking_mode():
    result = agent_with(ThinkingDeepSeek).invoke({"messages": [("user", "Trail 500 in stock?")]})
    assert result["messages"][-1].text == "5 in stock."


def test_plain_chatdeepseek_breaks_the_documented_rule():
    # Why ThinkingDeepSeek exists: the second model call drops the reasoning.
    with pytest.raises(openai.BadRequestError):
        agent_with(ChatDeepSeek).invoke({"messages": [("user", "Trail 500 in stock?")]})


def test_settings_check_the_key_for_each_provider(monkeypatch):
    deepseek_only = dict(model="deepseek:deepseek-flash", fallback_model="none",
                         embeddings="none", openai_api_key="")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert Settings(**deepseek_only).problems() == [
        "DEEPSEEK_API_KEY is not set (needed for deepseek)"]
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert Settings(**deepseek_only).problems() == []  # no OpenAI key needed at all
    # Hybrid search still embeds with OpenAI, whatever the chat model is.
    assert Settings(**{**deepseek_only, "embeddings": "openai"}).problems() == [
        "OPENAI_API_KEY is not set (needed for openai)"]
    assert "Unknown provider" in Settings(**{**deepseek_only, "model": "acme:x1"}).problems()[0]


def test_capstone_works_with_keyword_search_only(tmp_path):
    settings = Settings(index_dir=tmp_path / "index", db_path=str(tmp_path / "t.sqlite"), database_url="",
                        staff_token=STAFF_TOKEN, embeddings="none")
    app = create_app(settings, model=ScriptedModel(), fallback_model=None, embeddings=None)
    with TestClient(app) as client:
        response = client.post("/api/chat", json={
            "message": "Do PW-BRK-230 pads fit the City e-bike?",
            "customer_email": "sam@example.com"})
        reply = "".join(d["text"] for e, d in read_events(response) if e == "token")
    assert "PW-BRK-230" in reply
    assert not (tmp_path / "index").exists()  # nothing was embedded
