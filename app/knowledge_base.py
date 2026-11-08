"""Loads the FAQ knowledge base and finds the most relevant entries for a question.

Retrieval uses TF-IDF scoring with cosine similarity, written in plain Python so the
project has no heavy ML dependencies. The retrieved entries are passed to the LLM as
context (Retrieval-Augmented Generation, RAG).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on",
    "or", "so", "that", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "am", "was", "not", "still",
    "should", "did", "have", "has", "there",
}

DEFAULT_FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.json"


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stop words and apply light stemming."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass(frozen=True)
class FaqEntry:
    id: str
    category: str
    question: str
    answer: str


@dataclass(frozen=True)
class SearchResult:
    entry: FaqEntry
    score: float


class KnowledgeBase:
    def __init__(self, entries: list[FaqEntry]):
        if not entries:
            raise ValueError("Knowledge base is empty")
        self.entries = entries
        # Questions count twice: they are the best signal of what an entry is about.
        docs = [tokenize(f"{e.question} {e.question} {e.category} {e.answer}") for e in entries]
        doc_freq = Counter(term for doc in docs for term in set(doc))
        n = len(docs)
        self._idf = {t: math.log((1 + n) / (1 + df)) + 1 for t, df in doc_freq.items()}
        self._vectors = [self._vectorize(doc) for doc in docs]

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_FAQ_PATH) -> "KnowledgeBase":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls([FaqEntry(**item) for item in raw])

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vec = {t: c * self._idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 3, min_score: float = 0.1) -> list[SearchResult]:
        qvec = self._vectorize(tokenize(query))
        if not qvec:
            return []
        scored = []
        for entry, dvec in zip(self.entries, self._vectors):
            score = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            if score >= min_score:
                scored.append(SearchResult(entry, round(score, 3)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
