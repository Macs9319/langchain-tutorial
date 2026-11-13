"""
Tutorial 19 — Advanced RAG
==========================

Tutorial 06 built the basic recipe: split, embed, retrieve the closest
chunks, answer. It works, but it has well-known weak spots:
  - fixed-size chunks cut sections in half and lose their headings
  - embeddings are fuzzy: they match *meaning* well but exact codes like
    "PW-BRK-230" poorly
  - follow-up questions ("how much is it?") make bad search queries
  - the closest chunks aren't always the most useful ones
This tutorial fixes each one, then measures whether it helped.

What you'll learn:
  1. Split by document structure and keep headings as context and metadata
  2. Save the index to disk so documents are embedded once, not every run
  3. Hybrid search: combine keyword and vector search
  4. Rewrite conversational questions into good search queries
  5. Re-rank candidates with the model, and measure retrieval quality

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).

Run:
  uv run 19_advanced_rag.py
"""

import hashlib
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

HERE = Path(__file__).parent
DATA = HERE / "data"
INDEX_DIR = HERE / ".rag_index"

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# Step 1: Split by structure, keep the headings
# ---------------------------------------------------------------------------
# MarkdownHeaderTextSplitter cuts at headings, so each chunk is one whole
# section, and records the headings above it as metadata. We also put that
# heading path at the top of the chunk's text ("contextual chunk headers"):
# a chunk that just says "Price: 18 EUR per pair" is useless on its own, but
# "catalog.md > Spare parts > PW-BRK-220 brake pads" makes it findable.
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "topic")],
)


def load_chunks() -> list[Document]:
    chunks = []
    for path in sorted(DATA.glob("*.md")):
        for doc in splitter.split_text(path.read_text(encoding="utf-8")):
            headings = [doc.metadata[k] for k in ("section", "topic") if k in doc.metadata]
            doc.metadata["source"] = path.name
            doc.page_content = " > ".join([path.name, *headings]) + "\n" + doc.page_content
            chunks.append(doc)
    return chunks


# ---------------------------------------------------------------------------
# Step 2: Save the index to disk
# ---------------------------------------------------------------------------
# Embedding every document on every run wastes time and money. Save the
# index once and reload it. The file name is a fingerprint of the documents,
# so editing any document automatically triggers a rebuild.
def fingerprint(chunks: list[Document]) -> str:
    text = "\n".join(c.page_content for c in chunks)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def get_store(chunks: list[Document]) -> InMemoryVectorStore:
    INDEX_DIR.mkdir(exist_ok=True)
    path = INDEX_DIR / f"{fingerprint(chunks)}.json"
    start = time.perf_counter()
    if path.exists():
        store = InMemoryVectorStore.load(str(path), embeddings)
        action = "loaded"
    else:
        store = InMemoryVectorStore(embeddings)
        store.add_documents(chunks)  # the only step that calls the embeddings API
        store.dump(str(path))
        action = "built and saved"
    print(f"index {action} in {time.perf_counter() - start:.2f}s ({len(chunks)} chunks)")
    return store


# ---------------------------------------------------------------------------
# Step 3: Hybrid search
# ---------------------------------------------------------------------------
# Keyword search (here BM25, the classic search-engine formula) finds exact
# words and codes; vector search finds meaning. Run both and merge the two
# rankings with Reciprocal Rank Fusion (RRF): each chunk scores 1/(60 + rank)
# in every list it appears in. Chunks ranked well by both rise to the top.
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())  # keeps "pw-brk-220" whole


class KeywordIndex:
    """A minimal BM25 keyword index."""

    def __init__(self, chunks: list[Document], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tokens = [tokenize(c.page_content) for c in chunks]
        self.avg_len = sum(map(len, self.tokens)) / len(self.tokens)
        doc_freq = Counter(term for toks in self.tokens for term in set(toks))
        n = len(chunks)
        # Rare words matter more: "pw-brk-230" says far more than "the".
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in doc_freq.items()}

    def search(self, query: str, k: int = 5) -> list[Document]:
        scores = []
        for chunk, toks in zip(self.chunks, self.tokens):
            counts, score = Counter(toks), 0.0
            for term in set(tokenize(query)):
                tf = counts[term]
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            scores.append((score, chunk))
        ranked = sorted(scores, key=lambda pair: pair[0], reverse=True)
        return [chunk for score, chunk in ranked[:k] if score > 0]


def rrf(*rankings: list[Document], k: int = 60) -> list[Document]:
    """Merge several ranked lists into one (Reciprocal Rank Fusion)."""
    scores: dict[str, float] = {}
    by_text: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (k + rank)
            by_text[doc.page_content] = doc
    return [by_text[text] for text in sorted(scores, key=scores.get, reverse=True)]


class Retriever:
    def __init__(self, chunks: list[Document]):
        self.store = get_store(chunks)
        self.keywords = KeywordIndex(chunks)

    def vector(self, query: str, k: int = 5, source: str | None = None) -> list[Document]:
        # A metadata filter narrows the search to one document (from Step 1).
        only = (lambda d: d.metadata["source"] == source) if source else None
        return self.store.similarity_search(query, k=k, filter=only)

    def hybrid(self, query: str, k: int = 5) -> list[Document]:
        return rrf(self.vector(query, k=10), self.keywords.search(query, k=10))[:k]


def label(doc: Document) -> str:
    return doc.page_content.splitlines()[0]  # the heading path from Step 1


# ---------------------------------------------------------------------------
# Step 4: Query rewriting
# ---------------------------------------------------------------------------
# "What tyres does it take?" is a fine question in a conversation and a
# terrible search query: "it" matches nothing. Ask the model to turn the
# conversation into a few standalone queries, search each, and fuse them.
class SearchQueries(BaseModel):
    """Search queries for the latest question."""

    queries: list[str] = Field(description="2-3 short, standalone search queries that "
                                           "together cover the question. Resolve words like "
                                           "'it' or 'that one' using the conversation.")


def rewrite(conversation: list[tuple[str, str]]) -> list[str]:
    history = "\n".join(f"{role}: {text}" for role, text in conversation)
    result = llm.with_structured_output(SearchQueries).invoke(
        f"Write search queries for the user's LAST message.\n\nConversation:\n{history}")
    return result.queries


def multi_query_search(retriever: Retriever, queries: list[str], k: int = 5) -> list[Document]:
    return rrf(*(retriever.hybrid(q, k=10) for q in queries))[:k]


# ---------------------------------------------------------------------------
# Step 5: Re-ranking
# ---------------------------------------------------------------------------
# Search is fast but rough. Fetch more candidates than you need (say 8), let
# the model judge how well each one answers the question, and keep the best.
# Dedicated re-ranking models do this cheaper and faster; an LLM shows the idea.
class Relevance(BaseModel):
    scores: list[int] = Field(description="One score per passage, in order: 0 = irrelevant, "
                                          "10 = directly answers the question")


def rerank(question: str, candidates: list[Document], top_n: int = 3) -> list[Document]:
    passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(candidates))
    result = llm.with_structured_output(Relevance).invoke(
        f"Question: {question}\n\nScore each passage for how well it helps answer the "
        f"question.\n\n{passages}")
    scores = result.scores + [0] * (len(candidates) - len(result.scores))  # if it gave too few
    ranked = sorted(zip(scores, range(len(candidates))), reverse=True)
    return [candidates[i] for _, i in ranked[:top_n]]


def answer(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    return llm.invoke([
        ("system", "Answer using ONLY this context. If it isn't there, say you don't "
                   f"know. Two sentences at most.\n\nContext:\n{context}"),
        ("human", question),
    ]).text


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
# Questions with the section that holds the answer, to measure retrieval:
# "hit" = the right section is among the top 3 results.
TEST_SET = [
    ("Do PW-BRK-230 pads fit the City e-bike?", "PW-BRK-230 brake pads"),
    ("Which pads are quieter?", "PW-BRK-220 brake pads"),
    ("How far can the electric bike go on one charge?", "City e-bike"),
    ("How often should I replace the chain on my mountain bike?", "PW-CHN-12 chain"),
    ("What does a spare e-bike battery cost?", "PW-BAT-500 battery"),
    ("Can I get my money back after riding the bike?", "Return window"),
    ("Is the e-bike motor covered and for how long?", "Electric bikes"),
    ("Which bike suits a 7 year old?", "Kids 20"),
]


def hit(docs: list[Document], expected: str) -> bool:
    return any(expected in label(d) for d in docs[:3])


def main() -> None:
    print("=== Steps 1-2: structured chunks and a saved index ===")
    chunks = load_chunks()
    retriever = Retriever(chunks)
    example = next(c for c in chunks if "PW-BRK-220" in label(c))
    print("example chunk:\n  " + example.page_content.replace("\n", "\n  "))
    print("\nfiltered search (warranty.md only) for 'battery':")
    for d in retriever.vector("battery", k=2, source="warranty.md"):
        print("  ", label(d))

    print("\n=== Step 3: vector vs keyword vs hybrid ===")
    query = "Do PW-BRK-230 pads fit the City e-bike?"
    print("query:", query)
    for name, results in [("vector ", retriever.vector(query, k=3)),
                          ("keyword", retriever.keywords.search(query, k=3)),
                          ("hybrid ", retriever.hybrid(query, k=3))]:
        print(f"  {name}: " + " | ".join(label(d).split(" > ")[-1] for d in results))

    print("\n=== Step 4: query rewriting ===")
    conversation = [("user", "I'm thinking about the Gravel 300."),
                    ("assistant", "Great choice for mixed roads and trails!"),
                    ("user", "What tyres does it take, and what do spares cost?")]
    queries = rewrite(conversation)
    print("rewritten queries:", queries)
    for d in multi_query_search(retriever, queries, k=3):
        print("  ", label(d))

    print("\n=== Step 5: re-ranking, and measuring it all ===")
    totals = {"vector": 0, "hybrid": 0, "hybrid + rerank": 0}
    for question, expected in TEST_SET:
        candidates = retriever.hybrid(question, k=8)
        results = {"vector": retriever.vector(question, k=3),
                   "hybrid": candidates,
                   "hybrid + rerank": rerank(question, candidates)}
        marks = []
        for name, docs in results.items():
            totals[name] += hit(docs, expected)
            marks.append("✓" if hit(docs, expected) else "✗")
        print(f"  {' '.join(marks)}  {question}")
    print("  (columns: " + ", ".join(totals) + ")")
    for name, score in totals.items():
        print(f"  {name:<16} {score}/{len(TEST_SET)} questions found the right section")

    question = TEST_SET[0][0]
    print(f"\nfinal answer to '{question}':")
    print(" ", answer(question, rerank(question, retriever.hybrid(question, k=8))))


if __name__ == "__main__":
    main()
