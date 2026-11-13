"""End-to-end tests of the server, agent, tools and middleware, with a
scripted model (see conftest.py)."""

from pedalwise import shop

from .conftest import STAFF_TOKEN, read_events

SAM, DANA = "sam@example.com", "dana@example.com"
STAFF = {"X-Staff-Token": STAFF_TOKEN}


def chat(client, message, customer=SAM, thread_id=None):
    response = client.post("/api/chat", json={"message": message, "customer_email": customer,
                                              "thread_id": thread_id})
    return response, (read_events(response) if response.status_code == 200 else [])


def reply_text(events) -> str:
    return "".join(data["text"] for event, data in events if event == "token")


def test_pages_and_customers(client):
    assert client.get("/").status_code == 200
    assert client.get("/staff").status_code == 200
    assert {c["name"] for c in client.get("/api/customers").json()} == {"Sam", "Dana"}


def test_policy_question_streams_an_answer(client):
    response, events = chat(client, "Do PW-BRK-230 pads fit the City e-bike?")
    assert response.headers["content-type"].startswith("text/event-stream")
    kinds = [event for event, _ in events]
    assert kinds[0] == "thread" and kinds[-1] == "done"
    assert "status" in kinds  # "Looking that up…"
    assert "PW-BRK-230" in reply_text(events)  # the search found the right part


def test_tools_only_show_the_customers_own_orders(client):
    _, events = chat(client, "What are my orders?", customer=DANA)
    text = reply_text(events)
    assert "A1002" in text and "A1004" in text
    assert "A1001" not in text  # Sam's order


def test_stock_check_reports_progress(client):
    _, events = chat(client, "Is the Trail 500 in stock in Porto?")
    statuses = [data["text"] for event, data in events if event == "status"]
    assert any("warehouse" in s for s in statuses)  # from get_stream_writer
    assert "5 x Trail 500 in Porto" in reply_text(events)


def test_unknown_customer_is_rejected(client):
    response, _ = chat(client, "hi", customer="mallory@example.com")
    assert response.status_code == 401


def test_conversations_are_private(client):
    _, events = chat(client, "What are my orders?")
    thread_id = events[0][1]["thread_id"]
    # Sam can read it; Dana gets the same answer as for a missing conversation.
    assert client.get(f"/api/threads/{thread_id}", params={"customer_email": SAM}).status_code == 200
    assert client.get(f"/api/threads/{thread_id}", params={"customer_email": DANA}).status_code == 404
    response, _ = chat(client, "And now?", customer=DANA, thread_id=thread_id)
    assert response.status_code == 404


def test_refund_waits_for_staff_then_pays(client):
    _, events = chat(client, "Please refund order A1001, it's too small.")
    thread_id = events[0][1]["thread_id"]
    assert "approval_pending" in [event for event, _ in events]
    assert shop.REFUNDS == []  # nothing paid yet

    # The customer can't keep chatting in that conversation while it waits.
    response, _ = chat(client, "Hello?", thread_id=thread_id)
    assert response.status_code == 409

    # Staff endpoints need the token.
    assert client.get("/api/staff/approvals").status_code == 401
    queue = client.get("/api/staff/approvals", headers=STAFF).json()
    assert [item["thread_id"] for item in queue] == [thread_id]
    assert queue[0]["actions"][0]["args"]["order_id"] == "A1001"

    result = client.post(f"/api/staff/approvals/{thread_id}", headers=STAFF,
                         json={"approve": True}).json()
    assert "899.00" in result["reply"]
    assert shop.REFUNDS == [{"order_id": "A1001", "customer": SAM, "amount_eur": 899.0}]
    assert client.get("/api/staff/approvals", headers=STAFF).json() == []

    thread = client.get(f"/api/threads/{thread_id}", params={"customer_email": SAM}).json()
    assert thread["pending"] is False


def test_rejected_refund_pays_nothing(client):
    _, events = chat(client, "Refund A1002 please", customer=DANA)
    thread_id = events[0][1]["thread_id"]
    client.post(f"/api/staff/approvals/{thread_id}", headers=STAFF,
                json={"approve": False, "note": "Please send a photo first."})
    assert shop.REFUNDS == []


def test_ineligible_refund_never_reaches_staff(client):
    _, events = chat(client, "Refund order A1003 please")  # still in transit
    assert "approval_pending" not in [event for event, _ in events]
    assert "hasn't been delivered" in reply_text(events)
    assert client.get("/api/staff/approvals", headers=STAFF).json() == []
