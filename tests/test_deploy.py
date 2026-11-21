"""Tests for the deployment changes in Tutorial 21: health checks, settings
validation, structured logs and an approval queue that survives restarts."""

import json
import logging

from fastapi.testclient import TestClient

from pedalwise.config import Settings
from pedalwise.logs import JsonFormatter
from pedalwise.server import create_app

from .conftest import (STAFF_TOKEN, ScriptedModel, WordEmbeddings, read_events, refunds,
                       make_settings)

STAFF = {"X-Staff-Token": STAFF_TOKEN}
GOOD_TOKEN = "x" * 32


def make_app(tmp_path, db_url, **overrides):
    return create_app(make_settings(tmp_path, db_url, **overrides), model=ScriptedModel(),
                      fallback_model=None,
                      embeddings=WordEmbeddings())


def test_health_checks(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


def test_api_docs_only_in_development(client, tmp_path):
    assert client.get("/docs").status_code == 200
    db = str(tmp_path / "prod.sqlite")
    with TestClient(make_app(tmp_path, db, environment="production")) as prod:
        assert prod.get("/docs").status_code == 404
        assert prod.get("/openapi.json").status_code == 404
        assert prod.get("/healthz").status_code == 200


def test_pending_refund_survives_a_restart(tmp_path, db_url):
    with TestClient(make_app(tmp_path, db_url)) as first:
        response = first.post("/api/chat", json={"message": "Refund order A1001 please",
                                                 "customer_email": "sam@example.com"})
        thread_id = read_events(response)[0][1]["thread_id"]
    # A new process, same database: like a container being replaced.
    with TestClient(make_app(tmp_path, db_url)) as second:
        queue = second.get("/api/staff/approvals", headers=STAFF).json()
        assert [item["thread_id"] for item in queue] == [thread_id]
        second.post(f"/api/staff/approvals/{thread_id}", headers=STAFF, json={"approve": True})
        assert refunds(second) == [{"order_id": "A1001", "customer": "sam@example.com",
                                    "amount_eur": 899.0}]
        assert second.post(f"/api/staff/approvals/{thread_id}", headers=STAFF,
                           json={"approve": True}).status_code == 404  # can't decide twice


def test_settings_problems():
    # Explicit models, so a CHAT_MODEL in your .env doesn't change the test.
    openai = dict(model="gpt-5-mini", fallback_model="gpt-5-nano", embeddings="openai")
    assert Settings(**openai, openai_api_key="sk-test").problems() == []  # development is lenient
    assert Settings(**openai, openai_api_key="").problems() == [
        "OPENAI_API_KEY is not set (needed for openai)"]

    prod = dict(openai, environment="production", openai_api_key="sk-test")
    assert Settings(**prod, staff_token=GOOD_TOKEN).problems() == []
    assert Settings(**prod, staff_token="").problems() == ["STAFF_TOKEN must be set in production"]
    assert "at least 24" in Settings(**prod, staff_token="short").problems()[0]
    assert Settings(**openai, environment="prod", openai_api_key="sk-test").problems()


def test_settings_read_the_environment_when_created(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("STAFF_TOKEN", GOOD_TOKEN)
    settings = Settings()
    assert settings.port == 9000 and settings.staff_token == GOOD_TOKEN
    assert not settings.staff_token_generated
    assert GOOD_TOKEN not in repr(settings)  # secrets stay out of logs


def test_json_logs_include_extra_fields():
    record = logging.LogRecord("pedalwise", logging.INFO, __file__, 1, "Refund %s", ("decided",),
                               None)
    record.thread_id = "t-1"
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "Refund decided"
    assert line["thread_id"] == "t-1" and line["level"] == "INFO"
