"""Hybrid search over the shop's documents (Tutorial 19, as a reusable class)."""

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class KnowledgeBase:
    """Heading-aware chunks, a saved vector index, BM25 keywords, merged with RRF."""

    def __init__(self, data_dir: Path, index_dir: Path, embeddings: Embeddings):
        self.chunks = self._load_chunks(data_dir)
        self.store = self._load_or_build_index(index_dir, embeddings)
        self._build_keyword_index()

    @staticmethod
    def _load_chunks(data_dir: Path) -> list[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "topic")])
        chunks = []
        for path in sorted(data_dir.glob("*.md")):
            for doc in splitter.split_text(path.read_text(encoding="utf-8")):
                headings = [doc.metadata[k] for k in ("section", "topic") if k in doc.metadata]
                doc.metadata["source"] = path.name
                doc.page_content = " > ".join([path.name, *headings]) + "\n" + doc.page_content
                chunks.append(doc)
        return chunks

    def _load_or_build_index(self, index_dir: Path, embeddings: Embeddings) -> InMemoryVectorStore:
        text = "\n".join(c.page_content for c in self.chunks)
        path = index_dir / f"{hashlib.sha256(text.encode()).hexdigest()[:12]}.json"
        if path.exists():
            return InMemoryVectorStore.load(str(path), embeddings)
        store = InMemoryVectorStore(embeddings)
        store.add_documents(self.chunks)
        index_dir.mkdir(parents=True, exist_ok=True)
        store.dump(str(path))
        return store

    def _build_keyword_index(self) -> None:
        self._tokens = [_tokenize(c.page_content) for c in self.chunks]
        self._avg_len = sum(map(len, self._tokens)) / len(self._tokens)
        freq = Counter(t for toks in self._tokens for t in set(toks))
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in freq.items()}

    def keyword_search(self, query: str, k: int = 10) -> list[Document]:
        k1, b, terms = 1.5, 0.75, set(_tokenize(query))
        scored = []
        for chunk, toks in zip(self.chunks, self._tokens):
            counts = Counter(toks)
            score = sum(self._idf[t] * counts[t] * (k1 + 1)
                        / (counts[t] + k1 * (1 - b + b * len(toks) / self._avg_len))
                        for t in terms if counts[t])
            if score > 0:
                scored.append((score, chunk))
        return [c for _, c in sorted(scored, key=lambda p: p[0], reverse=True)[:k]]

    def search(self, query: str, k: int = 4) -> list[Document]:
        rankings = [self.store.similarity_search(query, k=10), self.keyword_search(query)]
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (60 + rank)
                docs[doc.page_content] = doc
        return [docs[t] for t in sorted(scores, key=scores.get, reverse=True)[:k]]
