"""Tests for Tutorial 22: run summaries, customer feedback and staff metrics."""

import uuid

from .conftest import STAFF_TOKEN, read_events

SAM, DANA = "sam@example.com", "dana@example.com"
STAFF = {"X-Staff-Token": STAFF_TOKEN}


def ask(client, message, customer=SAM):
    response = client.post("/api/chat", json={"message": message, "customer_email": customer})
    events = read_events(response)
    run_id = next(data["run_id"] for event, data in events if event == "run")
    return run_id, events


def metrics(client, **params):
    return client.get("/api/staff/metrics", headers=STAFF, params=params).json()


def test_every_chat_turn_is_recorded(client):
    ask(client, "What are my orders?")
    ask(client, "Is the Trail 500 in stock in Porto?")
    report = metrics(client)
    assert report["runs"] == 2 and report["outcomes"] == {"ok": 2}
    assert report["tools"] == {"my_orders": 1, "check_stock": 1}
    # Each turn: one call to pick a tool, one to answer. 100 + 10 tokens each.
    assert report["model_calls_per_run"] == 2
    assert report["tokens"] == {"input": 400, "output": 40}
    assert report["latency_ms"]["p50"] is not None


def test_refunds_record_the_approval_and_the_staff_decision(client):
    _, events = ask(client, "Refund order A1001 please")
    thread_id = events[0][1]["thread_id"]
    client.post(f"/api/staff/approvals/{thread_id}", headers=STAFF, json={"approve": True})
    report = metrics(client)
    assert report["outcomes"] == {"approval": 1, "ok": 1}
    assert report["tools"]["request_refund"] == 1


def test_feedback_is_saved_and_can_be_changed(client):
    run_id, _ = ask(client, "Do you sell skateboards?")
    vote = {"run_id": run_id, "customer_email": SAM}
    assert client.post("/api/feedback", json={**vote, "score": 1}).status_code == 200
    assert metrics(client)["feedback"] == {"up": 1, "down": 0, "rated_share": 1.0}
    client.post("/api/feedback", json={**vote, "score": -1, "comment": "Didn't answer"})
    report = metrics(client)
    assert report["feedback"]["down"] == 1 and report["feedback"]["up"] == 0
    [bad] = report["negative_feedback"]
    assert bad["question"] == "Do you sell skateboards?"
    assert bad["comment"] == "Didn't answer" and bad["answer"]


def test_feedback_only_on_your_own_runs(client):
    run_id, _ = ask(client, "What are my orders?")
    stranger = {"run_id": run_id, "customer_email": DANA, "score": -1}
    assert client.post("/api/feedback", json=stranger).status_code == 404
    unknown = {"run_id": str(uuid.uuid4()), "customer_email": SAM, "score": 1}
    assert client.post("/api/feedback", json=unknown).status_code == 404
    bad_score = {"run_id": run_id, "customer_email": SAM, "score": 5}
    assert client.post("/api/feedback", json=bad_score).status_code == 422


def test_metrics_need_the_staff_token(client):
    assert client.get("/api/staff/metrics").status_code == 401
    assert metrics(client)["runs"] == 0
