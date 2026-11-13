"""Pedalwise Bikes support app: the Tutorial 20 capstone.

Each module has one job, and most map to an earlier tutorial:

    config.py     settings from the environment
    shop.py       orders, stock and the refund rules, enforced in code (18)
    knowledge.py  hybrid search over the policy documents (06, 19)
    tools.py      what the agent can do, scoped to the current customer (05, 13, 15)
    agent.py      create_agent with middleware: approval, limits, retries,
                  fallbacks, PII masking, summarization (08, 13, 16, 18)
    server.py     FastAPI with streaming, saved conversations and a staff
                  approval queue (07, 11, 12)
    static/       the customer chat page and the staff page
"""
