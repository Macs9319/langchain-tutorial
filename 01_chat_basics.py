"""
Tutorial 01 — Your first LangChain chat query
=============================================

What you'll learn:
  1. Create a chat model (ChatOpenAI, backed by an OpenAI GPT model)
  2. Send a single prompt and read the reply
  3. Send a conversation using message objects (system + human)
  4. Inspect the response metadata (token usage, stop reason)
  5. Stream the reply token by token

Setup:
  uv sync                                   # install dependencies
  echo 'OPENAI_API_KEY=sk-...' > .env      # or export it in your shell
  # optional: OPENAI_MODEL=gpt-5  (defaults to gpt-5-mini)

Run:
  uv run 01_chat_basics.py
  uv run 01_chat_basics.py "Explain recursion in one sentence"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Loads OPENAI_API_KEY (and optional OPENAI_MODEL) from a .env file, if present.
load_dotenv()

# ---------------------------------------------------------------------------
# Step 1: Create the chat model
# ---------------------------------------------------------------------------
# A "chat model" in LangChain takes a list of messages and returns a message.
# Every provider (OpenAI, Anthropic, ...) exposes the same interface, so the
# rest of this script would work unchanged with a different model class.
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


def simple_query(question: str) -> None:
    """Step 2: The simplest call — pass a plain string."""
    print("=== Step 2: simple query ===")
    response = llm.invoke(question)  # returns an AIMessage
    print(response.text, "\n")


def conversation_query(question: str) -> None:
    """Step 3 & 4: Use message objects to add a system prompt."""
    print("=== Step 3: system + human messages ===")
    messages = [
        SystemMessage("You are a friendly tutor. Answer in at most 3 short bullet points."),
        HumanMessage(question),
    ]
    response = llm.invoke(messages)
    print(response.text, "\n")

    print("=== Step 4: response metadata ===")
    print("model       :", response.response_metadata.get("model_name"))
    print("stop reason :", response.response_metadata.get("finish_reason"))
    print("token usage :", response.usage_metadata, "\n")


def streaming_query(question: str) -> None:
    """Step 5: Stream the answer as it is generated."""
    print("=== Step 5: streaming ===")
    for chunk in llm.stream(question):
        print(chunk.text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is LangChain, in two sentences?"
    print(f"Question: {question}\n")

    simple_query(question)
    conversation_query(question)
    streaming_query(question)
