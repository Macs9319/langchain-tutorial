"""Tests for Tutorial 24's guardrails. No model: these check the database
rules hold whatever SQL arrives, which is the point of putting them there."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("text_to_sql", ROOT / "24_text_to_sql.py")
t24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t24)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("sales") / "sales.sqlite"
    t24.build_database(path)
    return path


@pytest.fixture
def guarded(db_path):
    return t24.GuardedDatabase(db_path)


def test_the_data_is_the_same_on_every_machine(db_path):
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM orders").fetchone() == (900,)
        assert db.execute("SELECT ROUND(SUM(quantity * unit_price_eur)) FROM order_items"
                          ).fetchone()[0] > 0


def test_ordinary_queries_work(guarded):
    columns, rows = guarded.query(
        "WITH t AS (SELECT store_id, COUNT(*) AS n FROM orders GROUP BY store_id) "
        "SELECT SUM(n) FROM t WHERE store_id IN (SELECT id FROM stores WHERE city LIKE 'L%')")
    assert rows[0][0] > 0


@pytest.mark.parametrize("sql, message", [
    ("DELETE FROM orders", "only SELECT"),
    ("UPDATE products SET price_eur = 0", "only SELECT"),
    ("SELECT 1; DROP TABLE orders", "one statement"),
    ("SELECT email FROM customers", "customers.email is prohibited"),
    ("SELECT * FROM customers", "customers.email is prohibited"),
    ("SELECT (SELECT email FROM customers LIMIT 1)", "customers.email is prohibited"),
    ("WITH e AS (SELECT email FROM customers) SELECT * FROM e", "customers.email is prohibited"),
    ("SELECT sql FROM sqlite_master", "prohibited"),
    ("SELECT name FROM pragma_table_info('customers')", "prohibited"),
    ("PRAGMA query_only = OFF", "only SELECT"),
    ("ATTACH DATABASE ':memory:' AS x", "only SELECT"),
    ("SELECT load_extension('evil')", "load_extension"),
])
def test_blocked(guarded, sql, message):
    with pytest.raises(sqlite3.Error, match=message):
        guarded.query(sql)
    assert "error" in guarded.history[-1]


def test_runaway_query_is_stopped(guarded, monkeypatch):
    monkeypatch.setattr(t24, "TIME_LIMIT_SECONDS", 0.2)
    with pytest.raises(sqlite3.Error, match="longer than"):
        guarded.query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
                      "SELECT COUNT(*) FROM n")


def test_big_results_are_cut(guarded):
    text = t24.format_rows(*guarded.query("SELECT * FROM order_items"))
    assert text.count("\n") == t24.MAX_ROWS + 1  # header + rows + note
    assert "only the first" in text


def test_tools_hide_private_columns(db_path, guarded):
    tools = {t.name: t for t in t24.make_tools(guarded, db_path)}
    described = tools["describe_table"].invoke({"table": "customers"})
    assert "email TEXT (hidden" in described and "@example.com" not in described
    assert tools["run_query"].invoke({"sql": "DELETE FROM orders"}).startswith("Error:")
    assert "orders: 900 rows" in tools["list_tables"].invoke({})


def test_grading_compares_values_not_sql():
    assert t24.rows_match([("Online", 236, 15.3)], [("online",)])      # extra columns are fine
    assert t24.rows_match([(1234.567,)], [(1234.57,)])                 # rounding is fine
    assert not t24.rows_match([("Porto",), ("Online",)], [("Online",)])  # row count matters
    assert not t24.rows_match([(100,)], [(120,)])
