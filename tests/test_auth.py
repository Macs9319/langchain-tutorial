"""Tests for Tutorial 27: logins, sessions, CSRF, throttling and the audit log.

Each test states one property an attacker would try to break.
"""

from fastapi.testclient import TestClient

from pedalwise.server import SESSION_COOKIE, create_app

from .conftest import (DEMO_PASSWORD, ScriptedModel, WordEmbeddings, login, make_settings,
                       read_events)

SAM, DANA, STAFF = "sam@example.com", "dana@example.com", "staff@example.com"


def db_rows(client, sql, params=()):
    db = client.app.state.users._db
    return client.portal.call(db.fetchall, sql, params)


def test_passwords_and_session_ids_are_never_stored(client):
    login(client, SAM)
    [(stored,)] = db_rows(client, "SELECT password_hash FROM users WHERE email = ?", (SAM,))
    assert stored.startswith("$argon2id$") and DEMO_PASSWORD not in stored
    cookie = client.cookies[SESSION_COOKIE]
    stored_sessions = [row[0] for row in db_rows(client, "SELECT token_hash FROM sessions")]
    assert cookie not in stored_sessions and len(stored_sessions) == 1  # only its hash


def test_wrong_email_and_wrong_password_look_the_same(client):
    unknown = client.post("/api/login", json={"email": "nobody@example.com", "password": "x" * 12})
    wrong = client.post("/api/login", json={"email": SAM, "password": "x" * 12})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"detail": "Wrong email or password"}


def test_guessing_is_throttled(client):
    for _ in range(5):
        assert login(client, DANA, password="wrong-password-1").status_code == 401
    # Locked, even with the right password: the guesser can't tell when they hit it.
    assert login(client, DANA).status_code == 429
    assert login(client, SAM).status_code == 200  # other accounts are unaffected


def test_cookie_flags(client, tmp_path, db_url):
    response = client.post("/api/login", json={"email": SAM, "password": DEMO_PASSWORD})
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert "secure" not in cookie  # development, plain http
    prod_settings = make_settings(tmp_path, str(tmp_path / "prod.sqlite"), environment="production")
    app = create_app(prod_settings, model=ScriptedModel(), fallback_model=None,
                     embeddings=WordEmbeddings())
    with TestClient(app) as prod:
        response = prod.post("/api/login", json={"email": SAM, "password": DEMO_PASSWORD})
        assert "secure" in response.headers["set-cookie"].lower()  # HTTPS only


def test_changes_need_the_csrf_token(client):
    login(client, SAM)
    token = client.headers.pop("X-CSRF-Token")
    assert client.post("/api/chat", json={"message": "What are my orders?"}).status_code == 403
    assert client.get("/api/me").status_code == 200  # reading is fine without it
    client.headers["X-CSRF-Token"] = "forged"
    assert client.post("/api/chat", json={"message": "What are my orders?"}).status_code == 403
    client.headers["X-CSRF-Token"] = token
    assert client.post("/api/chat", json={"message": "What are my orders?"}).status_code == 200


def test_logout_ends_the_session_on_the_server(client):
    login(client, SAM)
    stolen = client.cookies[SESSION_COOKIE]
    client.post("/api/logout")
    # Even someone who copied the cookie before logout can't use it now.
    client.cookies.set(SESSION_COOKIE, stolen)
    assert client.get("/api/me").status_code == 401


def test_the_customer_comes_from_the_session_not_the_request(client):
    login(client, SAM)
    # Trying to act as Dana by naming her: the field is simply ignored.
    response = client.post("/api/chat", json={"message": "What are my orders?",
                                              "customer_email": DANA})
    text = "".join(d["text"] for e, d in read_events(response) if e == "token")
    assert "A1001" in text and "A1002" not in text


def test_staff_decisions_are_audited_by_name(client):
    login(client, SAM)
    response = client.post("/api/chat", json={"message": "Refund order A1001 please"})
    thread_id = read_events(response)[0][1]["thread_id"]
    decide = f"/api/staff/approvals/{thread_id}"
    assert client.post(decide, json={"approve": True}).status_code == 403  # customers can't

    login(client, STAFF)
    assert client.post(decide, json={"approve": True}).status_code == 200
    [entry] = client.get("/api/staff/audit").json()
    assert entry["actor"] == STAFF and entry["action"] == "refund_approved"
    assert entry["detail"]["orders"] == ["A1001"] and entry["detail"]["customer"] == SAM
