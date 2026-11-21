"""
Tutorial 24 — Text-to-SQL: questions about your data, answered with queries
==========================================================================

Documents (Tutorials 06 and 19) answer "what's the policy?". Questions like
"which bikes sold best in Porto last month?" need numbers from a database.
An agent can answer them by writing SQL: it reads the schema, writes a
query, runs it, reads the result and explains it.

That means running SQL a model wrote, on your data. The model can misread
a question, write a query that scans forever, select columns it shouldn't
see, or be talked into DELETE by text hidden in the data itself. So most
of this tutorial is the guardrails, and they live in the database
connection rather than in the prompt or in checks on the SQL text:

  1. A read-only connection: writes fail, whatever the SQL says.
  2. SQLite's authorizer: a callback the database asks before it touches
     any table, column or function. Only allowed tables, no customer
     emails, a short list of functions, no PRAGMA or ATTACH. Checking the
     SQL text instead is easy to get wrong ("SELECT" can hide in comments,
     strings and subqueries); the authorizer sees what will actually run.
  3. A time limit and a row cap, so a runaway join can't hang the app or
     flood the model's context.
  4. Errors go back to the model, which usually fixes its own query.
  5. Every query is shown, so a person can check how a number was made.

The assistant is for Pedalwise staff. Letting customers query a database
needs more: each customer must only ever see their own rows (row-level
security, or the per-customer tools of Tutorial 20), never "just a prompt".

What you'll learn:
  1. Give an agent schema tools and a query tool (Tutorials 05 and 13)
  2. Enforce read-only access, allowed tables and hidden columns in the database
  3. Stop runaway queries with a progress handler and a row cap
  4. Evaluate text-to-SQL by comparing result rows, not SQL text (Tutorial 10)

Setup:
  Same as Tutorial 01. Uses CHAT_MODEL if set (Tutorial 23), e.g.
  CHAT_MODEL=deepseek:deepseek-flash. The first run creates sales.sqlite, a
  made-up sales database for the shop.

Run:
  uv run 24_text_to_sql.py                 # three example questions
  uv run 24_text_to_sql.py --chat          # ask your own
  uv run 24_text_to_sql.py --guardrails    # try to break the database (no API calls)
  uv run 24_text_to_sql.py --eval          # score the agent on 6 questions
"""

import argparse
import math
import os
import random
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

from pedalwise.models import chat_model

load_dotenv()

HERE = Path(__file__).parent
DB_PATH = HERE / "sales.sqlite"
TODAY = date(2026, 11, 20)  # the data's "today", so answers don't drift over time


# ---------------------------------------------------------------------------
# Step 1: A sales database to ask about
# ---------------------------------------------------------------------------
# Made up, but shaped like a real shop's: stores, products, customers,
# orders with line items, and stock. A fixed random seed makes it identical
# on every machine, so the evaluation's expected answers stay right.
SCHEMA = """
CREATE TABLE stores    (id INTEGER PRIMARY KEY, city TEXT NOT NULL);
CREATE TABLE products  (id INTEGER PRIMARY KEY, sku TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                        category TEXT NOT NULL, price_eur REAL NOT NULL);
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
                        city TEXT NOT NULL, joined_on TEXT NOT NULL);
CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers,
                        store_id INTEGER NOT NULL REFERENCES stores, ordered_at TEXT NOT NULL,
                        status TEXT NOT NULL);
CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders,
                          product_id INTEGER NOT NULL REFERENCES products,
                          quantity INTEGER NOT NULL, unit_price_eur REAL NOT NULL);
CREATE TABLE stock     (product_id INTEGER NOT NULL REFERENCES products,
                        store_id INTEGER NOT NULL REFERENCES stores, quantity INTEGER NOT NULL);
"""

# One line per table for the model: what a column name alone doesn't say.
TABLE_NOTES = {
    "stores": "The shops. 'Online' is the web shop.",
    "products": "category is 'bike', 'part' or 'accessory'.",
    "customers": "email is private and can't be queried.",
    "orders": "ordered_at is 'YYYY-MM-DD'. status: 'delivered', 'returned' or 'in transit'.",
    "order_items": "One row per product in an order. Revenue = quantity * unit_price_eur.",
    "stock": "Units in stock now, per product and store.",
}

PRODUCTS = [  # sku, name, category, price (matches data/catalog.md)
    ("BIKE-TRAIL-500", "Trail 500", "bike", 899), ("BIKE-GRAVEL-300", "Gravel 300", "bike", 1099),
    ("BIKE-CITY-E", "City e-bike", "bike", 2199), ("BIKE-KIDS-20", "Kids 20", "bike", 279),
    ("PW-BRK-220", "PW-BRK-220 brake pads", "part", 18),
    ("PW-BRK-230", "PW-BRK-230 brake pads", "part", 22),
    ("PW-BAT-500", "PW-BAT-500 battery", "part", 649), ("PW-CHN-12", "PW-CHN-12 chain", "part", 35),
    ("PW-TYR-45", "PW-TYR-45 tyre", "part", 42), ("HLM-COMMUTER", "Commuter helmet", "accessory", 59),
]
FIRST = ["Ana", "João", "Sam", "Dana", "Inês", "Rui", "Marta", "Tiago", "Sofia", "Pedro",
         "Leonor", "Miguel", "Beatriz", "Diogo", "Carla", "Nuno"]
LAST = ["Silva", "Santos", "Ferreira", "Pereira", "Costa", "Oliveira", "Rodrigues", "Martins",
        "Sousa", "Gomes"]


def build_database(path: Path) -> None:
    rng = random.Random(24)
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO stores VALUES (?, ?)", [(1, "Lisbon"), (2, "Porto"), (3, "Online")])
    db.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)",
                   [(i, *p) for i, p in enumerate(PRODUCTS, start=1)])

    customers = []
    for i in range(1, 81):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        joined = date(2024, 1, 1) + timedelta(days=rng.randrange(1000))
        customers.append((i, f"{first} {last}", f"{first.lower()}.{last.lower()}{i}@example.com",
                          rng.choice(["Lisbon", "Porto", "Braga", "Coimbra"]), joined.isoformat()))
    # Data is untrusted too. This name is a classic SQL injection; it's
    # harmless here because it's data, and the connection can't write.
    customers[41] = (42, "Robert'); DROP TABLE orders; --", "bobby@example.com", "Porto",
                     "2025-03-01")
    db.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    weights = [4, 3, 2, 2, 8, 6, 1, 6, 5, 7]  # parts and helmets sell more often than bikes
    order_id = 0
    for _ in range(900):
        order_id += 1
        day = date(2025, 1, 1) + timedelta(days=rng.randrange((TODAY - date(2025, 1, 1)).days + 1))
        store = rng.choices([1, 2, 3], weights=[4, 3, 3])[0]
        returned = rng.random() < (0.12 if store == 3 else 0.05)
        status = ("in transit" if (TODAY - day).days < 5 else "returned" if returned
                  else "delivered")
        db.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                   (order_id, rng.randint(1, 70), store, day.isoformat(), status))
        for product in set(rng.choices(range(1, 11), weights=weights, k=rng.randint(1, 3))):
            quantity = 1 if PRODUCTS[product - 1][2] == "bike" else rng.randint(1, 3)
            db.execute("INSERT INTO order_items VALUES (?, ?, ?, ?)",
                       (order_id, product, quantity, PRODUCTS[product - 1][3]))
    db.executemany("INSERT INTO stock VALUES (?, ?, ?)", [
        (p, s, 0 if rng.random() < 0.2 else rng.randint(1, 12))
        for p in range(1, 11) for s in (1, 2, 3)])
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Step 2: The guarded connection
# ---------------------------------------------------------------------------
ALLOWED_TABLES = set(TABLE_NOTES)
HIDDEN_COLUMNS = {("customers", "email")}
ALLOWED_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "total", "round", "abs", "coalesce", "ifnull",
    "nullif", "iif", "lower", "upper", "length", "substr", "instr", "replace", "trim",
    "date", "datetime", "strftime", "julianday", "group_concat", "printf",
    "like", "glob",  # SQLite reports the LIKE and GLOB operators as functions
}
MAX_ROWS = 50
TIME_LIMIT_SECONDS = 2.0


class GuardedDatabase:
    """Runs model-written SQL with the database itself enforcing the rules."""

    def __init__(self, path: Path):
        # mode=ro: SQLite opens the file read-only. query_only: belt and braces.
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.db.execute("PRAGMA query_only = ON")
        # Every real table, so names the query makes up itself (CTEs such as
        # WITH totals AS (...)) can be told apart from tables it may not read.
        self.real_tables = {name for (name,) in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        self.db.set_authorizer(self._authorize)
        # Called every 1000 steps of the query engine: stop it past the deadline.
        self.db.set_progress_handler(lambda: time.monotonic() > self._deadline, 1000)
        self._deadline = 0.0
        self.history: list[dict] = []  # every query run, for display and evaluation

    def _authorize(self, action, arg1, arg2, db_name, source):
        """SQLite asks this before each table read, function call or statement."""
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:  # arg1 = table, arg2 = column
            if (arg1, arg2) in HIDDEN_COLUMNS:
                return sqlite3.SQLITE_DENY
            # Reads from a CTE arrive under the CTE's own name. That's safe to
            # allow: the CTE's data came from real tables, checked here too.
            # Deny real tables not on the list, and SQLite's own tables
            # (sqlite_master, pragma_table_info(...), dbstat...).
            if arg1 in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            if (arg1 in self.real_tables or arg1.startswith(("sqlite_", "pragma_"))
                    or arg1 == "dbstat"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:  # arg2 = function name
            return sqlite3.SQLITE_OK if arg2.lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY  # INSERT, DELETE, PRAGMA, ATTACH, CREATE...

    def query(self, sql: str) -> tuple[list[str], list[tuple]]:
        """Run one statement. Raises sqlite3.Error with a message the model can act on."""
        self._deadline = time.monotonic() + TIME_LIMIT_SECONDS
        try:
            cursor = self.db.execute(sql)  # one statement only: "a; b" is refused
            rows = cursor.fetchmany(MAX_ROWS + 1)
            columns = [c[0] for c in cursor.description or []]
        except sqlite3.Error as error:
            message = str(error)
            # Say what IS allowed, so the model can fix the query.
            if "interrupted" in message:
                message = f"query took longer than {TIME_LIMIT_SECONDS}s and was stopped"
            elif "prohibited" in message:
                message += " (use the tables from list_tables; hidden columns can't be read)"
            elif "to use function" in message:
                message += f" (allowed functions: {', '.join(sorted(ALLOWED_FUNCTIONS))})"
            elif "not authorized" in message:
                message += " (only SELECT queries are allowed)"
            self.history.append({"sql": sql, "error": message})
            raise sqlite3.Error(message) from None
        self.history.append({"sql": sql, "columns": columns, "rows": rows[:MAX_ROWS]})
        return columns, rows


def format_rows(columns: list[str], rows: list[tuple]) -> str:
    """A compact text table. Long results are cut so they can't flood the context."""
    lines = [" | ".join(columns)]
    for row in rows[:MAX_ROWS]:
        lines.append(" | ".join(str(v)[:80] for v in row))
    if len(rows) > MAX_ROWS:
        lines.append(f"(only the first {MAX_ROWS} rows shown: aggregate or add LIMIT)")
    if not rows:
        lines.append("(no rows)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3: Tools and the agent
# ---------------------------------------------------------------------------
def make_tools(guarded: GuardedDatabase, path: Path) -> list:
    # Schema tools use a separate, trusted connection: reading sqlite_master
    # is something we do, not something the model's SQL is allowed to do.
    def schema_db():
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    @tool
    def list_tables() -> str:
        """List the tables, with row counts and notes. Start here."""
        with schema_db() as db:
            return "\n".join(
                f"{t}: {db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows. {note}"
                for t, note in TABLE_NOTES.items())

    @tool
    def describe_table(table: str) -> str:
        """Show a table's columns and three example rows."""
        if table not in ALLOWED_TABLES:
            return f"Unknown table {table!r}. Tables: {', '.join(sorted(ALLOWED_TABLES))}"
        with schema_db() as db:
            columns = [(name, kind) for _, name, kind, *_ in db.execute(f"PRAGMA table_info({table})")]
            visible = [n for n, _ in columns if (table, n) not in HIDDEN_COLUMNS]
            sample = db.execute(f"SELECT {', '.join(visible)} FROM {table} LIMIT 3").fetchall()
        lines = [f"{n} {k}" + (" (hidden: can't be queried)" if (table, n) in HIDDEN_COLUMNS
                               else "") for n, k in columns]
        return f"{table}: {TABLE_NOTES[table]}\n" + "\n".join(lines) + \
            "\n\nExample rows:\n" + format_rows(visible, sample)

    @tool
    def run_query(sql: str) -> str:
        """Run one read-only SQLite SELECT and return the rows. On an error,
        read the message, fix the query and try again."""
        try:
            return format_rows(*guarded.query(sql))
        except sqlite3.Error as error:
            return f"Error: {error}"

    return [list_tables, describe_table, run_query]


SYSTEM_PROMPT = f"""You answer questions from Pedalwise Bikes staff using the shop's SQLite database.

- Today is {TODAY.isoformat()}. "Last month" means the calendar month before this one.
- Look at the tables you need (list_tables, describe_table) before writing SQL.
  Never guess column names.
- Write SQLite SQL. Aggregate in SQL instead of fetching many rows.
- If a query fails, fix it using the error message.
- Answer in one to three sentences with the numbers, then show the final
  query in a ```sql block so staff can check it.
- If the data can't answer the question, say so. Never invent numbers."""


def build_agent(guarded: GuardedDatabase, path: Path = DB_PATH):
    spec = os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return create_agent(chat_model(spec, max_tokens=4096, timeout=60),
                        tools=make_tools(guarded, path), system_prompt=SYSTEM_PROMPT)


def ask(agent, question: str, show_steps: bool = True) -> str:
    """Stream the agent's steps, printing each SQL query as it runs."""
    answer = ""
    for update in agent.stream({"messages": [("user", question)]}, stream_mode="updates"):
        for node in update.values():
            for message in (node or {}).get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    if show_steps and call["name"] == "run_query":
                        print(f"  ↳ SQL: {' '.join(call['args']['sql'].split())}")
                    elif show_steps:
                        print(f"  ↳ {call['name']}({call['args'] or ''})")
                if message.type == "tool" and message.text.startswith("Error") and show_steps:
                    print(f"    {message.text}")
                if message.type == "ai" and not message.tool_calls:
                    answer = message.text
    return answer


# ---------------------------------------------------------------------------
# Step 4: Try to break it (no model needed)
# ---------------------------------------------------------------------------
ATTACKS = [
    ("A normal query", "SELECT city, COUNT(*) AS orders FROM orders JOIN stores "
                       "ON stores.id = orders.store_id GROUP BY city"),
    ("Delete data", "DELETE FROM orders"),
    ("Two statements", "SELECT 1; DROP TABLE orders"),
    ("Read a hidden column", "SELECT name, email FROM customers LIMIT 3"),
    ("Hide it in a subquery", "SELECT (SELECT email FROM customers LIMIT 1)"),
    ("Read the schema table", "SELECT sql FROM sqlite_master"),
    ("Read it another way", "SELECT name FROM pragma_table_info('customers')"),
    ("Use a CTE (allowed)", "WITH totals AS (SELECT store_id, COUNT(*) AS n FROM orders "
                            "GROUP BY store_id) SELECT MAX(n) FROM totals"),
    ("Change a setting", "PRAGMA query_only = OFF"),
    ("Open another file", "ATTACH DATABASE 'memory.sqlite' AS other"),
    ("A function not on the list", "SELECT randomblob(10)"),
    ("A query that never ends", "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
                                "SELECT COUNT(*) FROM n"),
    ("Too many rows", "SELECT * FROM order_items"),
    ("Injection stored in the data", "SELECT name FROM customers WHERE id = 42"),
]


def guardrails_demo() -> None:
    guarded = GuardedDatabase(DB_PATH)
    for name, sql in ATTACKS:
        started = time.perf_counter()
        try:
            columns, rows = guarded.query(sql)
            result = f"allowed: {len(rows[:MAX_ROWS])} row(s)" + \
                (f", cut at {MAX_ROWS}" if len(rows) > MAX_ROWS else "") + f", e.g. {rows[0]}"
        except sqlite3.Error as error:
            result = f"blocked: {error}"
        print(f"{name:<30} {result[:90]}  ({time.perf_counter() - started:.2f}s)")
    with sqlite3.connect(DB_PATH) as db:  # prove nothing changed
        print(f"\nOrders still in the database: {db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]}")


# ---------------------------------------------------------------------------
# Step 5: Evaluation
# ---------------------------------------------------------------------------
# Compare what the agent's last query returned with what a reference query
# returns. Comparing SQL text would fail correct answers: there are many
# ways to write the same query. Each case names the reference SQL; the
# agent's rows must contain the reference values (extra columns are fine).
#
# Grading can be wrong too. Asked "which store is highest?", an agent may
# list every store, ranked: a good answer, but a different row count. So
# "which one" cases are graded on the agent's top row only. When a case
# fails, read what the agent returned before blaming the model.
EVAL_CASES = [
    ("How many orders were placed in Porto in October 2026?",
     "SELECT COUNT(*) FROM orders o JOIN stores s ON s.id = o.store_id "
     "WHERE s.city = 'Porto' AND o.ordered_at BETWEEN '2026-10-01' AND '2026-10-31'"),
    ("What were the top 3 products by revenue in 2026? Count every order.",
     "SELECT p.name, SUM(i.quantity * i.unit_price_eur) AS revenue FROM order_items i "
     "JOIN orders o ON o.id = i.order_id JOIN products p ON p.id = i.product_id "
     "WHERE o.ordered_at >= '2026-01-01' GROUP BY p.id ORDER BY revenue DESC LIMIT 3"),
    ("Which store has the highest share of orders returned?",
     "SELECT s.city FROM orders o JOIN stores s ON s.id = o.store_id GROUP BY s.id "
     "ORDER BY AVG(o.status = 'returned') DESC LIMIT 1", "top row"),
    ("How many customers have never placed an order?",
     "SELECT COUNT(*) FROM customers WHERE id NOT IN (SELECT customer_id FROM orders)"),
    ("What was the average value of an online order in 2026, in EUR?",
     "SELECT AVG(t) FROM (SELECT SUM(i.quantity * i.unit_price_eur) AS t FROM orders o "
     "JOIN order_items i ON i.order_id = o.id WHERE o.store_id = 3 "
     "AND o.ordered_at >= '2026-01-01' GROUP BY o.id)"),
    ("Which products are out of stock in Lisbon?",
     "SELECT p.name FROM stock k JOIN products p ON p.id = k.product_id "
     "WHERE k.store_id = 1 AND k.quantity = 0"),
]


def same_value(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.005, abs_tol=0.01)  # allow rounding
    return str(a).strip().casefold() == str(b).strip().casefold()


def rows_match(agent_rows: list[tuple], reference_rows: list[tuple]) -> bool:
    """Same number of rows, and each reference row's values appear in a distinct agent row."""
    if len(agent_rows) != len(reference_rows):
        return False
    unused = list(agent_rows)
    for ref in reference_rows:
        found = next((row for row in unused
                      if all(any(same_value(v, a) for a in row) for v in ref)), None)
        if found is None:
            return False
        unused.remove(found)
    return True


def evaluate() -> None:
    with sqlite3.connect(DB_PATH) as db:
        expected = [db.execute(case[1]).fetchall() for case in EVAL_CASES]
    passed = 0
    for (question, _, *mode), reference in zip(EVAL_CASES, expected):
        guarded = GuardedDatabase(DB_PATH)
        agent = build_agent(guarded)
        started = time.perf_counter()
        try:
            ask(agent, question, show_steps=False)
        except Exception as error:
            print(f"ERROR {question}\n      {error!r}")
            continue
        runs = [h for h in guarded.history if "rows" in h]
        got = runs[-1]["rows"] if runs else []
        ok = bool(runs) and rows_match(got[:1] if mode == ["top row"] else got, reference)
        passed += ok
        count, errors = len(guarded.history), sum("error" in h for h in guarded.history)
        print(f"{'PASS' if ok else 'FAIL'}  {question}\n      {count} quer{'y' if count == 1 else 'ies'}"
              + (f" ({errors} failed, then fixed)" if errors else "")
              + f", {time.perf_counter() - started:.1f}s")
        if not ok:
            print(f"      expected {reference}\n      got      "
                  f"{runs[-1]['rows'][:5] if runs else 'no successful query'}")
    print(f"\n{passed}/{len(EVAL_CASES)} correct")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the sales database questions in English.")
    parser.add_argument("--chat", action="store_true", help="ask your own questions")
    parser.add_argument("--guardrails", action="store_true", help="try to break it (no API calls)")
    parser.add_argument("--eval", action="store_true", help="score the agent on test questions")
    parser.add_argument("--rebuild", action="store_true", help="recreate sales.sqlite")
    args = parser.parse_args()

    if args.rebuild or not DB_PATH.exists():
        build_database(DB_PATH)
        print(f"Created {DB_PATH.name}\n")

    if args.guardrails:
        guardrails_demo()
    elif args.eval:
        evaluate()
    else:
        agent = build_agent(GuardedDatabase(DB_PATH))
        questions = ([] if args.chat else [
            "Which bike sold the most units in Porto last month?",
            "What share of online orders were returned in 2026?",
            "List our 3 best customers by total spend, with their emails.",
        ])
        for question in questions:
            print(f"\nQ: {question}")
            print(f"A: {ask(agent, question)}")
        while args.chat:
            question = input("\nQuestion (or quit): ").strip()
            if question.lower() in ("quit", "exit", ""):
                break
            print(f"A: {ask(agent, question)}")
