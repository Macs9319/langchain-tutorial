"""Tests for Tutorial 25's grading and decision rule, with the scripted model."""

import asyncio
import importlib.util
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from pedalwise import shop
from pedalwise.agent import build_agent
from pedalwise.knowledge import KnowledgeBase
from pedalwise.tools import make_tools

from .conftest import ScriptedModel, WordEmbeddings

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("model_selection", ROOT / "25_model_selection.py")
t25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t25)
CASES = {c.name: c for c in t25.CASES}


def run(case_name, tmp_path):
    shop.REFUNDS.clear()  # other tests may have refunded A1001 already
    kb = KnowledgeBase(ROOT / "data", tmp_path / "index", WordEmbeddings())
    agent = build_agent(ScriptedModel(), None, make_tools(kb), InMemorySaver())
    return asyncio.run(t25.run_case(agent, CASES[case_name], judge=None))


def test_behaviour_checks(tmp_path):
    assert run("my orders", tmp_path).passed
    assert run("eligible refund goes to staff", tmp_path).passed        # reached the queue
    assert run("someone else's order", tmp_path).passed                 # and this didn't


def test_curly_apostrophes_dont_fail_correct_answers():
    case = t25.Case("x", "x", must_include=[["hasn't been delivered"]])
    low = "it hasn’t been delivered".lower().replace("’", "'")
    assert all(any(o.lower() in low for o in opts) for opts in case.must_include)


def row(model, critical=1.0, quality=1.0, p50=2.0, cost=1.0):
    return {"model": model, "critical": critical, "quality": quality, "p50_s": p50,
            "usd_per_1000": cost}


def test_decision_rule(capsys):
    t25.recommend([row("unsafe", critical=0.9, cost=0.1), row("slow", p50=9, cost=0.2),
                   row("good", quality=0.95, cost=0.5), row("best", quality=1.0, cost=2.0),
                   row("weak", quality=0.7, cost=0.3)], max_seconds=5)
    out = capsys.readouterr().out
    assert "Recommendation: good" in out            # cheapest within 10 points of the best
    assert "critical case: unsafe" in out and "5s: slow" in out
