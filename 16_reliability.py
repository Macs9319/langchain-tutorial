"""
Tutorial 16 — Reliability: retries, fallbacks, rate limits and caching
=====================================================================

Every tutorial so far assumed each API call works. In production they
sometimes don't: the provider is overloaded (429), a request times out, the
network drops, a model is retired. This tutorial adds the safety nets that
keep an app working, and keeps costs down while doing it.

What you'll learn:
  1. Which errors are worth retrying, and the client's own timeout/retry settings
  2. Retry any step with .with_retry()
  3. Switch to a backup model with .with_fallbacks()
  4. Stay under rate limits with a rate limiter
  5. Skip repeated calls entirely with a cache
  6. The same protections for agents, as middleware

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).
  Optional: FALLBACK_MODEL (default gpt-5-nano), the backup model.

Run:
  uv run 16_reliability.py
"""

import asyncio
import os
import time

import openai
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.caches import InMemoryCache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-5-nano")


# ---------------------------------------------------------------------------
# Step 1: Know your errors
# ---------------------------------------------------------------------------
# Retrying only helps when the problem is temporary:
#   retry      RateLimitError (429), APITimeoutError, APIConnectionError,
#              InternalServerError (5xx): wait a moment and try again
#   don't      AuthenticationError (bad key), BadRequestError (bad input),
#              NotFoundError (unknown model): the same request will fail
#              again, so fix it or fall back to something else instead
RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    TimeoutError,
    ConnectionError,
)

# The OpenAI client already retries retryable errors with growing waits
# ("exponential backoff"). These two settings are your first line of defence:
#   timeout      give up on one attempt after this many seconds
#   max_retries  how many extra attempts after a retryable failure
llm = ChatOpenAI(model=MODEL, max_tokens=4096, timeout=60, max_retries=3)


# ---------------------------------------------------------------------------
# Step 2: Retry any step with .with_retry()
# ---------------------------------------------------------------------------
# .with_retry() works on any Runnable: a chain, a tool, your own function.
# Use it for steps the client's retries don't cover. Here, a fake service
# that fails twice before succeeding shows the mechanics without API costs.
attempts = {"count": 0}


def flaky_inventory_service(item: str) -> str:
    attempts["count"] += 1
    if attempts["count"] < 3:
        print(f"  attempt {attempts['count']}: connection failed")
        raise ConnectionError("inventory service unreachable")
    print(f"  attempt {attempts['count']}: ok")
    return f"{item}: 4 in stock"


def retry_demo() -> None:
    print("=== Step 2: with_retry ===")
    reliable = RunnableLambda(flaky_inventory_service).with_retry(
        retry_if_exception_type=RETRYABLE,  # never retry a bug like KeyError
        stop_after_attempt=4,               # 1 try + up to 3 retries
        wait_exponential_jitter=True,       # wait longer each time, plus randomness
        exponential_jitter_params={"multiplier": 0.2, "max": 2},  # first wait ~0.2s, cap 2s
    )
    print("result:", reliable.invoke("Trail 500"), "\n")
    # The random "jitter" matters when many requests fail at once: without
    # it they would all retry at the same moment and overload the service again.


# ---------------------------------------------------------------------------
# Step 3: Fallback to a backup model
# ---------------------------------------------------------------------------
# When retrying won't help (or retries run out), try something else. Here the
# primary model name doesn't exist, which fails with NotFoundError, so the
# request moves on to the fallback. The same thing happens during an outage
# or when a model is retired.
def fallback_demo() -> None:
    print("=== Step 3: with_fallbacks ===")
    primary = ChatOpenAI(model="model-that-does-not-exist", max_retries=0)
    backup = ChatOpenAI(model=FALLBACK_MODEL, max_tokens=4096)

    robust_llm = primary.with_fallbacks([backup])  # list: tried in order
    reply = robust_llm.invoke("Say 'hello from the backup model' and nothing else.")
    print("reply      :", reply.text)
    print("answered by:", reply.response_metadata.get("model_name"), "\n")
    # Fallbacks can be anything with the same input and output: another
    # provider (e.g. ChatAnthropic), a cheaper model, or even a fixed
    # "Sorry, please try again later" reply as the last resort.


# ---------------------------------------------------------------------------
# Step 4: Rate limiting
# ---------------------------------------------------------------------------
# Providers cap how many requests you may send per minute. Instead of hitting
# the limit and getting 429s, slow yourself down first. The limiter is a
# "token bucket": it refills at requests_per_second, and each call takes one.
async def rate_limit_demo() -> None:
    print("=== Step 4: rate limiting ===")
    limiter = InMemoryRateLimiter(
        requests_per_second=1,      # long-run average
        check_every_n_seconds=0.1,  # how often a waiting call checks again
        max_bucket_size=1,          # no bursts: strictly one at a time
    )
    limited_llm = ChatOpenAI(model=MODEL, max_tokens=4096, rate_limiter=limiter)

    start = time.perf_counter()
    questions = [f"Reply with just the number {n}." for n in range(1, 5)]
    # abatch would normally send all 4 at once; the limiter spaces them out.
    replies = await limited_llm.abatch(questions)
    print(f"answers: {[r.text.strip() for r in replies]}")
    print(f"4 calls took {time.perf_counter() - start:.1f}s (at least ~3s at 1 request/s)\n")
    # One limiter shared by every model object keeps the *whole app* under the
    # limit. It only works within one process; several servers need a shared
    # limiter (e.g. in Redis).


# ---------------------------------------------------------------------------
# Step 5: Caching
# ---------------------------------------------------------------------------
# If the exact same request was answered before, return the saved answer:
# instant, and free. Great for development and for repeated questions; not
# for anything that should vary between calls.
def cache_demo() -> None:
    print("=== Step 5: caching ===")
    cached_llm = ChatOpenAI(model=MODEL, max_tokens=4096, cache=InMemoryCache())
    question = "In one sentence, why do bicycles need regular servicing?"

    for label in ["first call ", "second call"]:
        start = time.perf_counter()
        reply = cached_llm.invoke(question)
        print(f"{label}: {time.perf_counter() - start:5.2f}s  {reply.text[:70]}...")
    print("(the second answer came from the cache, with no API call)\n")
    # The key is the whole request: model, settings and every message. Change
    # one character and it's a miss. InMemoryCache is lost on exit; for a
    # cache that survives restarts, see SQLiteCache in langchain-community.


# ---------------------------------------------------------------------------
# Step 6: Agents
# ---------------------------------------------------------------------------
# create_agent (Tutorial 13) gets the same protections from middleware, and
# they apply to every model call the agent makes, however many steps it takes.
def agent_demo() -> None:
    print("=== Step 6: retry and fallback middleware for agents ===")
    agent = create_agent(
        ChatOpenAI(model="model-that-does-not-exist", max_retries=0),
        system_prompt="You are Pedalwise Bikes support. One short sentence.",
        # Order matters: each middleware wraps the ones listed after it.
        # Fallback is outermost, so the inner retry gets its chances first:
        #   primary (try, retry, retry) -> all failed -> fallback model
        # In the opposite order, the fallback would catch the first failure
        # and the retries would never happen.
        middleware=[
            ModelFallbackMiddleware(ChatOpenAI(model=FALLBACK_MODEL, max_tokens=4096)),
            # Retry each model call up to 2 more times, waiting 1s, then 2s
            # (doubling), with jitter. Here the model doesn't exist, so the
            # retries fail too. on_failure="error" then re-raises the error so
            # the fallback can take over; the default ("continue") would
            # instead reply to the user with the error text and stop there.
            ModelRetryMiddleware(max_retries=2, initial_delay=1.0, backoff_factor=2.0,
                                 on_failure="error"),
        ],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Are you open on Sundays?"}]})
    reply = result["messages"][-1]
    print("reply      :", reply.text)
    print("answered by:", reply.response_metadata.get("model_name"), "\n")


async def main() -> None:
    retry_demo()
    fallback_demo()
    await rate_limit_demo()
    cache_demo()
    agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
