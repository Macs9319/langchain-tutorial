"""
Tutorial 06 — RAG: answering questions from your own documents
==============================================================

A model only knows what it saw in training. It has never seen your company
handbook, so asking it about your returns policy gets a guess. RAG
(retrieval-augmented generation) fixes this: find the passages of your
documents that are relevant to the question, paste them into the prompt,
and ask the model to answer from them.

The documents in data/ describe "Pedalwise Bikes", a made-up bike shop.

What you'll learn:
  1. Load files as Document objects
  2. Split them into small chunks with a text splitter
  3. Turn chunks into embeddings and store them in a vector store
  4. Retrieve the chunks most similar to a question
  5. Build a RAG chain: retrieve -> prompt -> model -> answer, with sources

Setup:
  Same as Tutorial 01 (OPENAI_API_KEY in .env, optional OPENAI_MODEL).
  This tutorial also calls OpenAI's embeddings API.

Run:
  uv run 06_rag.py
  uv run 06_rag.py "How long is the battery warranty on e-bikes?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Step 1: Load documents
# ---------------------------------------------------------------------------
# A Document is text plus metadata. The metadata (here, the file name) stays
# attached through every later step, which lets us cite sources at the end.
def load_documents() -> list[Document]:
    return [
        Document(page_content=path.read_text(encoding="utf-8"),
                 metadata={"source": path.name})
        for path in sorted(DATA_DIR.glob("*.md"))
    ]


# ---------------------------------------------------------------------------
# Step 2: Split into chunks
# ---------------------------------------------------------------------------
# Whole files are too coarse: we want to retrieve just the relevant passage.
# RecursiveCharacterTextSplitter splits on paragraphs first, then lines, then
# words, so chunks break at natural boundaries. The overlap repeats a little
# text between neighbouring chunks so a sentence cut in half isn't lost.
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


# ---------------------------------------------------------------------------
# Step 3: Embed and store
# ---------------------------------------------------------------------------
# An embedding is a list of numbers that captures what a text *means*.
# Texts about similar things get similar numbers, even with different words
# ("refund" vs "money back"). The vector store keeps each chunk next to its
# embedding so it can find the closest ones to a question.
# InMemoryVectorStore is lost when the script exits; real apps use a
# persistent store (Chroma, pgvector, Pinecone, ...) with the same interface.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store() -> InMemoryVectorStore:
    print("=== Steps 1-3: load, split, embed ===")
    docs = load_documents()
    chunks = splitter.split_documents(docs)
    print(f"loaded {len(docs)} files -> {len(chunks)} chunks")
    print(f"example chunk from {chunks[0].metadata['source']}:\n"
          f"  {chunks[0].page_content[:120]!r}...")

    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)  # embeds every chunk (one API call per batch)
    print()
    return store


# ---------------------------------------------------------------------------
# Step 4: Retrieve
# ---------------------------------------------------------------------------
def show_retrieval(store: InMemoryVectorStore, question: str) -> None:
    print("=== Step 4: retrieval ===")
    print("question:", question)
    # Score = cosine similarity: closer to 1 means more similar in meaning.
    for doc, score in store.similarity_search_with_score(question, k=3):
        first_line = doc.page_content.strip().splitlines()[0]
        print(f"  {score:.3f}  [{doc.metadata['source']}] {first_line}")
    print()


# ---------------------------------------------------------------------------
# Step 5: The RAG chain
# ---------------------------------------------------------------------------
# The prompt tells the model to answer only from the provided context and to
# admit when the answer isn't there. Without that instruction it will happily
# fill gaps with plausible-sounding guesses.
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You answer customer questions for Pedalwise Bikes using ONLY the "
         "context below. If the context doesn't contain the answer, say you "
         "don't know. Keep answers to 2-3 sentences.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one string, labelled with their source."""
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


def build_rag_chain(store: InMemoryVectorStore):
    # A retriever is a Runnable: question string in, list of Documents out.
    retriever = store.as_retriever(search_kwargs={"k": 3})

    # The dict runs both branches on the same input (the question):
    #   context  -> retriever finds chunks -> format_docs joins them
    #   question -> RunnablePassthrough passes the question through unchanged
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    ), retriever


def ask(store: InMemoryVectorStore, questions: list[str]) -> None:
    print("=== Step 5: RAG answers ===")
    rag_chain, retriever = build_rag_chain(store)
    for question in questions:
        answer = rag_chain.invoke(question)
        # Retrieve again just to show which files the answer came from.
        sources = sorted({d.metadata["source"] for d in retriever.invoke(question)})
        print(f"Q: {question}\nA: {answer}\n   sources: {', '.join(sources)}\n")


if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "Can I return a bike I've already ridden?",
        "How long is the battery warranty on e-bikes?",
        "Is the workshop open on Sunday?",
        "Do you sell skateboards?",  # not in the documents: should say "don't know"
    ]

    vector_store = build_vector_store()
    show_retrieval(vector_store, questions[0])
    ask(vector_store, questions)
