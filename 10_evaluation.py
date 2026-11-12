"""
Tutorial 10 — Evaluating your app
=================================

Every tutorial so far ended with "run it and look at the answer". That
doesn't scale: change the prompt or the retriever and an answer that used
to be right may quietly break. An *evaluation* fixes this: a fixed set of
test questions with known good answers, run automatically, scored, and
compared across versions of your app.

We evaluate the RAG chain from Tutorial 06 in three configurations to see
which settings actually matter.

What you'll learn:
  1. Write a small test set: questions + reference answers + key facts
  2. Make the app configurable so variants can be compared
  3. Grade answers two ways: simple rule checks and an LLM judge
  4. Track token usage (cost) per variant with a callback
  5. Compare variants in a score table and inspect the failures

Optional — tracing with LangSmith:
  LangChain can record every step of every run (prompts, retrieved chunks,
  tool calls, timings) to LangSmith, a hosted dashboard. This sends your
  prompts and data to LangSmith's servers, so it is off unless you turn it
  on. To enable it, create an account at smith.langchain.com and add to .env:
      LANGSMITH_TRACING=true
      LANGSMITH_API_KEY=lsv2_...
  No code changes are needed; every tutorial in this series will then trace.

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 10_evaluation.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

HERE = Path(__file__).parent

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)
# The judge grades the answers. Ideally use a model at least as capable as
# the one being tested; set JUDGE_MODEL in .env to try a stronger one.
judge_llm = ChatOpenAI(
    model=os.getenv("JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini")),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Step 1: The test set
# ---------------------------------------------------------------------------
# Each case has:
#   reference    - what a correct answer says (used by the LLM judge)
#   must_include - key facts for the rule check. Each inner list holds
#                  acceptable spellings; at least one must appear.
# Include questions the documents DON'T answer: a good app says "I don't
# know" instead of inventing something. Real test sets come from real user
# questions and grow every time you find a bug.
TEST_SET = [
    {"question": "Can I return a bike I've already ridden?",
     "reference": "Yes, within 14 days, with a 10% restocking fee.",
     "must_include": [["14 days", "14-day", "fourteen"], ["10%", "10 %", "ten percent"]]},
    {"question": "How long are e-bike batteries covered?",
     "reference": "2 years or 500 full charge cycles, whichever comes first.",
     "must_include": [["2 years", "two years", "2-year"], ["500"]]},
    {"question": "How much is a full service?",
     "reference": "120 EUR, with parts extra.",
     "must_include": [["120"]]},
    {"question": "Is the workshop open on Sundays?",
     "reference": "No. Workshops are closed on Sundays and public holidays.",
     "must_include": [["closed", "not open"]]},  # not "no": it matches inside "know"
    {"question": "How quickly will I get my refund?",
     "reference": "Within 5 business days of the item reaching the warehouse; "
                  "store credit is instant with a 5% bonus.",
     "must_include": [["5 business days", "five business days", "5 working days"]]},
    {"question": "Is crash damage covered by the frame warranty?",
     "reference": "No, but owners get 40% off a replacement frame.",
     "must_include": [["40%", "40 %", "forty percent"]]},
    {"question": "Do you sell skateboards?",
     "reference": "The documents don't say; the assistant should say it doesn't know.",
     "must_include": []},  # rule check can't judge this one; the LLM judge does
    {"question": "What's the store's phone number?",
     "reference": "The documents don't include a phone number; the assistant "
                  "should say it doesn't know (mentioning the warranty email is fine).",
     "must_include": []},
]


# ---------------------------------------------------------------------------
# Step 2: The app under test, made configurable
# ---------------------------------------------------------------------------
# Same RAG chain as Tutorial 06, but the settings we want to compare are
# parameters. The vector store is built once and shared by every variant.
def build_store() -> InMemoryVectorStore:
    docs = [Document(p.read_text(encoding="utf-8"), metadata={"source": p.name})
            for p in sorted((HERE / "data").glob("*.md"))]
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
    store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents(chunks)
    return store


STRICT_SYSTEM = ("You answer customer questions for Pedalwise Bikes using ONLY the "
                 "context below. If the context doesn't contain the answer, say you "
                 "don't know. Keep answers to 2-3 sentences.\n\nContext:\n{context}")
LOOSE_SYSTEM = ("You are a helpful assistant for Pedalwise Bikes. Here is some "
                "context that may help:\n{context}")


def build_rag_chain(store: InMemoryVectorStore, k: int, system_prompt: str):
    retriever = store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)  # noqa: E731
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser())


VARIANTS = {
    "k=3, strict prompt": {"k": 3, "system_prompt": STRICT_SYSTEM},  # Tutorial 06 setup
    "k=1, strict prompt": {"k": 1, "system_prompt": STRICT_SYSTEM},  # less context
    "k=3, loose prompt":  {"k": 3, "system_prompt": LOOSE_SYSTEM},   # no "don't know" rule
}


# ---------------------------------------------------------------------------
# Step 3: Graders
# ---------------------------------------------------------------------------
def rule_check(answer: str, must_include: list[list[str]]) -> bool | None:
    """Pass if every key fact appears in some accepted spelling.

    Fast, free and predictable, but brittle: it can't recognise a correct
    answer worded differently, or judge whether "I don't know" was right.
    Returns None when the case has no key facts to check.
    """
    if not must_include:
        return None
    text = answer.lower()
    return all(any(option.lower() in text for option in options) for options in must_include)


class Grade(BaseModel):
    """The judge's verdict on one answer."""

    reason: str = Field(description="One sentence explaining the verdict")
    correct: bool = Field(description="True if the answer agrees with the reference "
                                      "and adds no false or invented information")


judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are grading a customer support assistant. Compare the ANSWER "
               "with the REFERENCE. Mark it correct only if it gives the same facts "
               "and adds nothing false or made up. Wording and length don't matter. "
               "If the reference says the assistant should not know, the answer is "
               "correct only if it declines rather than inventing an answer."),
    ("human", "QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}"),
])
# Reasoning comes before the verdict in Grade, so the judge explains first
# and decides second, which tends to give more reliable grades.
judge = judge_prompt | judge_llm.with_structured_output(Grade)


# ---------------------------------------------------------------------------
# Step 4 & 5: Run every variant, score it, compare
# ---------------------------------------------------------------------------
def evaluate_variant(store: InMemoryVectorStore, settings: dict) -> dict:
    chain = build_rag_chain(store, **settings)
    questions = [case["question"] for case in TEST_SET]

    # Every model call inside this block reports its token usage to `usage`.
    # Only the app's calls are inside, so the judge's tokens aren't counted.
    with get_usage_metadata_callback() as usage:
        answers = chain.batch(questions)  # all questions in parallel

    grades = judge.batch([
        {"question": case["question"], "reference": case["reference"], "answer": answer}
        for case, answer in zip(TEST_SET, answers)
    ])

    rows = []
    for case, answer, grade in zip(TEST_SET, answers, grades):
        rows.append({"question": case["question"], "answer": answer,
                     "rule": rule_check(answer, case["must_include"]),
                     "judge": grade.correct, "judge_reason": grade.reason})

    tokens = sum(u["total_tokens"] for u in usage.usage_metadata.values())
    return {"rows": rows, "tokens": tokens}


def score(values: list[bool | None]) -> str:
    checked = [v for v in values if v is not None]
    return f"{sum(checked)}/{len(checked)}" if checked else "-"


def main() -> None:
    print(f"Building the document index, then running {len(TEST_SET)} test "
          f"questions x {len(VARIANTS)} variants...\n")
    store = build_store()
    results = {name: evaluate_variant(store, settings) for name, settings in VARIANTS.items()}

    print("=== Step 5: scores ===")
    print(f"{'variant':<22} {'rule check':>10} {'LLM judge':>10} {'app tokens':>11}")
    for name, result in results.items():
        rows = result["rows"]
        print(f"{name:<22} {score([r['rule'] for r in rows]):>10} "
              f"{score([r['judge'] for r in rows]):>10} {result['tokens']:>11,}")

    # The score says *whether* something broke; the failures say *why*.
    print("\n=== Failures (judge said incorrect) ===")
    for name, result in results.items():
        for row in result["rows"]:
            if not row["judge"]:
                print(f"[{name}] {row['question']}\n"
                      f"   answer: {row['answer'][:150]}\n"
                      f"   judge : {row['judge_reason']}\n")

    print("Tip: with only 8 questions, a one-question difference may be luck. "
          "Rerun a few times, and grow the test set before trusting small gaps.")


if __name__ == "__main__":
    main()
