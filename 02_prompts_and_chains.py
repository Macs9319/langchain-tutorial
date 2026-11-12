"""
Tutorial 02 — Prompt templates and chains
=========================================

In Tutorial 01 we built message lists by hand. Real apps reuse the same
prompt with different inputs, and post-process the model's reply. LangChain
handles both with *prompt templates* and *chains*.

What you'll learn:
  1. Build a reusable prompt with ChatPromptTemplate and {placeholders}
  2. Turn the model's AIMessage into a plain string with StrOutputParser
  3. Connect prompt -> model -> parser into a chain with the `|` operator
  4. Run one chain over many inputs at once with .batch()
  5. Stream a chain's output with .stream()

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 02_prompts_and_chains.py
  uv run 02_prompts_and_chains.py "black holes"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)

# ---------------------------------------------------------------------------
# Step 1: A prompt template
# ---------------------------------------------------------------------------
# Each (role, text) pair becomes a message. Anything in {curly braces} is a
# variable you fill in later, so one template serves many requests.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a tutor who explains things to a {audience}. "
                   "Keep answers under {max_sentences} sentences."),
        ("human", "Explain {topic}."),
    ]
)

# ---------------------------------------------------------------------------
# Step 2: An output parser
# ---------------------------------------------------------------------------
# The model returns an AIMessage. StrOutputParser pulls out just the text.
parser = StrOutputParser()

# ---------------------------------------------------------------------------
# Step 3: A chain
# ---------------------------------------------------------------------------
# The `|` operator pipes the output of one step into the next:
#   dict of variables -> prompt (messages) -> llm (AIMessage) -> parser (str)
# This style is called LCEL (LangChain Expression Language). Every piece is
# a "Runnable", so the chain has .invoke(), .batch() and .stream() too.
chain = prompt | llm | parser


def show_template(topic: str) -> None:
    print("=== Step 1: what the template produces ===")
    filled = prompt.invoke({"audience": "10-year-old", "max_sentences": 3, "topic": topic})
    for message in filled.to_messages():
        print(f"[{message.type}] {message.content}")
    print()


def run_chain(topic: str) -> None:
    print("=== Step 3: invoke the chain ===")
    answer = chain.invoke({"audience": "10-year-old", "max_sentences": 3, "topic": topic})
    print(type(answer).__name__, "->", answer, "\n")  # a plain str, thanks to the parser


def run_batch(topic: str) -> None:
    print("=== Step 4: batch — same topic, different audiences ===")
    audiences = ["10-year-old", "university student", "domain expert"]
    inputs = [{"audience": a, "max_sentences": 2, "topic": topic} for a in audiences]
    # .batch() sends the requests in parallel and returns results in order.
    for audience, answer in zip(audiences, chain.batch(inputs)):
        print(f"--- {audience} ---\n{answer}\n")


def run_stream(topic: str) -> None:
    print("=== Step 5: stream the chain ===")
    for piece in chain.stream({"audience": "curious adult", "max_sentences": 4, "topic": topic}):
        print(piece, end="", flush=True)  # each piece is already a str
    print("\n")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "how the internet works"
    print(f"Topic: {topic}\n")

    show_template(topic)
    run_chain(topic)
    run_batch(topic)
    run_stream(topic)
