"""Magazyn wektorów z wyszukiwaniem gęstym, leksykalnym i hybrydowym.

Implementacja oparta na numpy, wystarczająca do dziesiątek tysięcy fragmentów.
Przy większej skali podmienia się to na FAISS albo bazę wektorową; interfejs
`search()` pozostaje ten sam.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


@dataclass(frozen=True)
class SearchResult:
    """Pojedynczy wynik wyszukiwania.

    Attributes:
        chunk: Znaleziony fragment.
        score: Wynik dopasowania. Wyżej = lepiej.
        rank: Pozycja w rankingu (od 1).
    """

    chunk: Chunk
    score: float
    rank: int

    def __str__(self) -> str:
        return f"#{self.rank} ({self.score:.3f}) {self.chunk}"


class BM25:
    """Wyszukiwanie leksykalne BM25, uzupełnienie dla wyszukiwania gęstego.

    Embeddingi gubią rzadkie terminy: nazwy własne, numery, akronimy. BM25 je
    łapie, bo dopasowuje dosłownie. Połączenie obu metod daje wyraźnie lepsze
    wyniki niż każda z osobna, stąd tryb hybrydowy w VectorStore.

    Args:
        k1: Nasycenie częstością termu. Wyżej = częstość waży więcej.
        b: Wpływ długości dokumentu (0 = brak, 1 = pełna normalizacja).
    """

    TOKEN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._avg_length = 0.0
        self._idf: dict[str, float] = {}

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls.TOKEN.findall(text.lower())

    def fit(self, texts: list[str]) -> None:
        """Buduje indeks na podstawie korpusu."""
        self._docs = [Counter(self.tokenize(t)) for t in texts]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(doc.keys())

        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Zwraca wynik BM25 dla każdego dokumentu w korpusie."""
        if not self._docs:
            return np.zeros(0, dtype=np.float32)

        terms = self.tokenize(query)
        result = np.zeros(len(self._docs), dtype=np.float32)

        for i, doc in enumerate(self._docs):
            length = self._lengths[i]
            norm = (
                1 - self.b + self.b * length / self._avg_length
                if self._avg_length
                else 1.0
            )
            total = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                total += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            result[i] = total

        return result


def _min_max(values: np.ndarray) -> np.ndarray:
    """Skaluje wektor do zakresu [0, 1]. Potrzebne przed łączeniem wyników."""
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if math.isclose(hi, lo):
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class VectorStore:
    """Magazyn fragmentów z wyszukiwaniem gęstym, BM25 i hybrydowym.

    Przykład:
        >>> store = VectorStore(HashingEmbedder())
        >>> store.add(chunks)
        >>> for result in store.search("pytanie", top_k=3):
        ...     print(result)

    Args:
        embedder: Model embeddingowy.
        enable_bm25: Czy budować indeks leksykalny (potrzebny do trybu hybrid).
    """

    def __init__(self, embedder: Embedder, enable_bm25: bool = True) -> None:
        self.embedder = embedder
        self.enable_bm25 = enable_bm25
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._bm25 = BM25() if enable_bm25 else None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Dodaje fragmenty do magazynu i przelicza indeksy.

        Args:
            chunks: Fragmenty do zaindeksowania.
            batch_size: Rozmiar partii przy embedowaniu.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            new_vectors.append(self.embedder.embed(texts[i : i + batch_size]))
        stacked = np.vstack(new_vectors)

        self._chunks.extend(chunks)
        self._vectors = (
            stacked
            if self._vectors is None
            else np.vstack([self._vectors, stacked])
        )

        if self._bm25 is not None:
            self._bm25.fit([c.text for c in self._chunks])

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        alpha: float = 0.5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Wyszukuje fragmenty najlepiej pasujące do zapytania.

        Args:
            query: Zapytanie użytkownika.
            top_k: Ile wyników zwrócić.
            mode: "dense" (embeddingi), "lexical" (BM25) albo "hybrid" (oba).
            alpha: W trybie hybrydowym, waga wyszukiwania gęstego.
                1.0 = tylko embeddingi, 0.0 = tylko BM25.
            min_score: Odcina wyniki poniżej progu.

        Returns:
            Lista wyników posortowana malejąco po dopasowaniu.
        """
        if self._vectors is None or not self._chunks:
            return []

        if mode == "dense":
            scores = self._dense_scores(query)
        elif mode == "lexical":
            scores = self._lexical_scores(query)
        elif mode == "hybrid":
            dense = _min_max(self._dense_scores(query))
            lexical = _min_max(self._lexical_scores(query))
            scores = alpha * dense + (1 - alpha) * lexical
        else:
            raise ValueError(
                f"Nieznany tryb: {mode!r}. Dostępne: dense, lexical, hybrid"
            )

        # argpartition zamiast pełnego sortowania, przy dużym korpusie
        # znacząco szybsze.
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_idx, start=1):
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(
                SearchResult(chunk=self._chunks[idx], score=score, rank=rank)
            )

        return results

    def _dense_scores(self, query: str) -> np.ndarray:
        """Podobieństwo kosinusowe. Wektory są znormalizowane, więc to iloczyn."""
        assert self._vectors is not None
        query_vec = self.embedder.embed_one(query)
        return self._vectors @ query_vec

    def _lexical_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError(
                "Wyszukiwanie leksykalne wyłączone (enable_bm25=False)"
            )
        return self._bm25.scores(query)

    def save(self, path: str | Path) -> None:
        """Zapisuje indeks na dysk.

        Embedder nie jest serializowany, przy wczytywaniu trzeba podać ten sam.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "vectors": self._vectors,
                    "enable_bm25": self.enable_bm25,
                    "dimension": self.embedder.dimension,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> VectorStore:
        """Wczytuje indeks z dysku.

        Raises:
            ValueError: Wymiar embeddera nie zgadza się z zapisanym indeksem.
        """
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)

        saved_dim = data.get("dimension")
        if saved_dim is not None and saved_dim != embedder.dimension:
            raise ValueError(
                f"Niezgodność wymiarów: indeks zapisano z embedderem "
                f"{saved_dim}-wymiarowym, podano {embedder.dimension}-wymiarowy."
            )

        store = cls(embedder, enable_bm25=data.get("enable_bm25", True))
        store._chunks = data["chunks"]
        store._vectors = data["vectors"]
        if store._bm25 is not None and store._chunks:
            store._bm25.fit([c.text for c in store._chunks])
        return store

    def stats(self) -> dict[str, Any]:
        """Podstawowe statystyki indeksu, przydatne przy diagnozie jakości."""
        if not self._chunks:
            return {"chunks": 0}

        lengths = [c.char_count for c in self._chunks]
        return {
            "chunks": len(self._chunks),
            "documents": len({c.doc_id for c in self._chunks}),
            "dimension": self.embedder.dimension,
            "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
            "min_chunk_chars": min(lengths),
            "max_chunk_chars": max(lengths),
            "bm25_enabled": self._bm25 is not None,
        }
