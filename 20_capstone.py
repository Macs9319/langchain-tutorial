"""
Tutorial 20 — Capstone: a complete support app
==============================================

The finale puts the series together into one app: a Pedalwise Bikes
support assistant that customers chat with in the browser, and that asks a
staff member before any refund. Unlike the single-file tutorials, the code
lives in a package, pedalwise/, split the way a real project would be:

    pedalwise/config.py     settings from .env
    pedalwise/shop.py       data and business rules, plain Python      (18)
    pedalwise/knowledge.py  hybrid search over data/                    (06, 19)
    pedalwise/tools.py      tools scoped to the signed-in customer      (05, 13, 15)
    pedalwise/agent.py      create_agent + middleware                   (08, 13, 16)
    pedalwise/server.py     FastAPI, streaming, saved conversations     (07, 11, 12)
    pedalwise/static/       chat page and staff page
    tests/                  tests that run without an API key           (10)

What the app does:
  - answers policy, product and part-number questions from the documents
  - lists the customer's own orders and checks stock, with live progress
  - checks refund eligibility in code, then pauses for staff approval
  - masks card numbers, caps runaway loops, retries and falls back on errors
  - saves conversations in SQLite so they survive restarts

What you'll learn:
  1. How to structure an LLM app so each part can be changed and tested alone
  2. How the pieces from Tutorials 01-19 fit together
  3. How to test an agent app without calling a model (tests/)

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env). Optional: STAFF_TOKEN, a fixed
  token for the staff page (otherwise a random one is printed at startup).

Run:
  uv run 20_capstone.py
  customers: http://127.0.0.1:8001          (pick Sam or Dana at the top)
  staff:     http://127.0.0.1:8001/staff    (paste the staff token)

Test (no API key or network needed):
  uv run pytest
"""

import uvicorn

from pedalwise.config import Settings
from pedalwise.server import create_app

if __name__ == "__main__":
    settings = Settings()
    # 127.0.0.1: reachable only from this computer. Opening it to a network
    # needs real logins first; anyone who can reach it can spend your credits.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
