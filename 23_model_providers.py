"""
Tutorial 23 — Model providers: adding DeepSeek
==============================================

Tutorials 01-22 all use OpenAI. Tying an app to one provider means one
expired key, outage or price rise stops everything. LangChain's chat models
share one interface (invoke, stream, bind_tools, with_structured_output),
so the provider can be a setting instead of code. This tutorial adds
DeepSeek as a second provider, compares it with OpenAI on the same tasks,
and runs the Pedalwise app on it.

What you'll learn:
  1. init_chat_model and "provider:model" strings: switching provider in config
  2. Comparing providers on the same tasks: correctness, speed, tokens
  3. Where providers differ underneath the shared interface:
       - DeepSeek "thinks" by default, and its docs say the reasoning must
         be sent back when tools are used. LangChain doesn't, and the API
         didn't object when tested, but pedalwise/models.py follows the docs:
         code to an API's contract, not just to what it lets you get away with
       - DeepSeek has no embeddings API, so search falls back to keywords
       - prices differ, and DeepSeek's change with the time of day
  4. Falling back from one provider to another when a call fails (Tutorial 16)

DeepSeek's models (see api-docs.deepseek.com for the current list):
  deepseek-flash     fast and cheap; also reads images
  deepseek-v4-pro    stronger, slower, about 3-4x the price
  Both handle tool calls and JSON output, with a 1M-token context.

Setup:
  1. Create a key at platform.deepseek.com and add to .env:
         DEEPSEEK_API_KEY=sk-...
  2. Optional, to compare: OPENAI_API_KEY as before. Providers without a
     key are skipped.

Run:
  uv run 23_model_providers.py                    # compare every provider you have a key for
  uv run 23_model_providers.py --models deepseek:deepseek-flash deepseek:deepseek-v4-pro
  uv run 23_model_providers.py --thinking         # watch DeepSeek reason before answering

The capstone on DeepSeek alone (no OpenAI key needed), in .env or inline:
  CHAT_MODEL=deepseek:deepseek-flash FALLBACK_MODEL=none EMBEDDINGS=none \\
      uv run 21_deployment.py
"""

import argparse
import os
import time

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from pedalwise.models import API_KEY_VARS, chat_model, split_spec

load_dotenv()

DEFAULT_MODELS = ["openai:gpt-5-mini", "deepseek:deepseek-flash", "deepseek:deepseek-v4-pro"]


def has_key(spec: str) -> bool:
    return bool(os.getenv(API_KEY_VARS.get(split_spec(spec)[0], ""), ""))


# ---------------------------------------------------------------------------
# Step 1: Three tasks every provider must handle
# ---------------------------------------------------------------------------
# The same three skills the series relies on: answering (01), structured
# output (03) and tool calling (05). A provider that fails one of these can't
# run the capstone, however good its plain answers are.
class FrameSize(BaseModel):
    """A bike frame size recommendation."""

    frame_cm: int = Field(description="Recommended road bike frame size in cm")
    reason: str = Field(description="One short sentence")


@tool
def check_stock(item: str, store: str) -> str:
    """Check stock of a bike at a Pedalwise store (Lisbon or Porto)."""
    return f"3 x {item} in {store}"


def task_answer(model):
    reply = model.invoke("In one sentence: what does a bike's derailleur do?")
    return bool(reply.text.strip()), reply.text, reply


def task_structured(model):
    # include_raw: also return the raw message, to read its token usage.
    result = model.with_structured_output(FrameSize, include_raw=True).invoke(
        "I'm 178 cm tall. Which road bike frame size should I get?")
    parsed = result["parsed"]
    ok = isinstance(parsed, FrameSize) and 45 <= parsed.frame_cm <= 65
    return ok, repr(parsed), result["raw"]


def task_tool_call(model):
    reply = model.bind_tools([check_stock]).invoke("Is the Trail 500 in stock in Porto?")
    calls = reply.tool_calls
    ok = bool(calls) and calls[0]["name"] == "check_stock" and \
        "porto" in str(calls[0]["args"]).lower()
    return ok, str(calls[0]["args"]) if calls else reply.text, reply


TASKS = {"answer": task_answer, "structured": task_structured, "tool call": task_tool_call}


# ---------------------------------------------------------------------------
# Step 2: Compare
# ---------------------------------------------------------------------------
def compare(specs: list[str]) -> list[str]:
    """Run every task on every model with a key. Returns the models that passed."""
    print(f"{'model':<28} {'task':<11} {'ok':<4} {'secs':>5} {'in':>6} {'out':>6}  result")
    passed = []
    for spec in specs:
        if not has_key(spec):
            var = API_KEY_VARS.get(split_spec(spec)[0], "an API key")
            print(f"{spec:<28} skipped: {var} is not set")
            continue
        model = chat_model(spec, max_tokens=2048, timeout=60, max_retries=1)
        all_ok = True
        for name, task in TASKS.items():
            started = time.perf_counter()
            try:
                ok, result, message = task(model)
                usage = message.usage_metadata or {}
            except Exception as error:  # an expired key, an unknown model...
                ok, result, usage = False, f"{type(error).__name__}: {str(error)[:80]}", {}
            all_ok &= ok
            print(f"{spec:<28} {name:<11} {'yes' if ok else 'NO':<4} "
                  f"{time.perf_counter() - started:>5.1f} {usage.get('input_tokens', '-'):>6} "
                  f"{usage.get('output_tokens', '-'):>6}  {' '.join(result.split())[:60]}")
        if all_ok:
            passed.append(spec)
    # Tokens are what you pay for, but the price per token differs a lot by
    # provider and model, and DeepSeek charges half outside its peak hours.
    # Multiply by the current prices on each provider's pricing page.
    return passed


# ---------------------------------------------------------------------------
# Step 3: DeepSeek's thinking mode
# ---------------------------------------------------------------------------
def thinking_demo() -> None:
    """Thinking: the model reasons first, and you pay for those tokens too."""
    if not has_key("deepseek:deepseek-flash"):
        print("Set DEEPSEEK_API_KEY to run this.")
        return
    question = ("A customer bought a bike 20 days ago and rode it twice. The return window "
                "is 30 days for unused bikes, 14 days for ridden ones. Can they return it?")
    for thinking in (False, True):
        model = chat_model("deepseek:deepseek-flash", deepseek_thinking=thinking, max_tokens=4096)
        started = time.perf_counter()
        reply = model.invoke(question)
        reasoning = reply.additional_kwargs.get("reasoning_content", "")
        usage = reply.usage_metadata or {}
        print(f"\nThinking {'ON' if thinking else 'OFF'}: {time.perf_counter() - started:.1f}s, "
              f"{usage.get('output_tokens')} output tokens")
        if reasoning:
            print(f"  reasoning ({len(reasoning)} chars): {' '.join(reasoning.split())[:200]}...")
        print(f"  answer: {reply.text}")
    print("\nThe reasoning is billed as output tokens. Turn it on (DEEPSEEK_THINKING=true)\n"
          "where answers need several steps of logic; leave it off for quick replies.\n"
          "With tools, DeepSeek's docs say the reasoning must go back on the next\n"
          "call: pedalwise/models.py's ThinkingDeepSeek does that.")


# ---------------------------------------------------------------------------
# Step 4: Falling back across providers
# ---------------------------------------------------------------------------
def fallback_demo(working: list[str]) -> None:
    """If the primary provider fails, try another (Tutorial 16, across providers)."""
    if not working:
        print("\nFallback demo skipped: no model passed all tasks.")
        return
    # A model name that doesn't exist stands in for an outage or a dead key.
    broken = chat_model("deepseek:no-such-model" if has_key("deepseek:deepseek-flash")
                        else "openai:no-such-model", max_retries=0)
    # Reasoning models (gpt-5-mini) spend tokens thinking before they answer:
    # too low a cap and the answer comes back empty.
    backup = chat_model(working[0], max_tokens=2048, timeout=60)
    model = broken.with_fallbacks([backup])
    reply = model.invoke("Say hello in five words.")
    print(f"\nPrimary failed, so {working[0]} answered: {reply.text}")
    print("In the capstone: CHAT_MODEL=deepseek:deepseek-flash with "
          "FALLBACK_MODEL=openai:gpt-5-nano, or the other way round.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare chat model providers.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help='models as "provider:model"')
    parser.add_argument("--thinking", action="store_true", help="DeepSeek thinking mode demo")
    args = parser.parse_args()
    if args.thinking:
        thinking_demo()
    else:
        passed = compare(args.models)
        fallback_demo(passed)
