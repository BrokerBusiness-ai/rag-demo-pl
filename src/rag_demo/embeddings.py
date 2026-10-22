"""Modele embeddingowe za wspólnym interfejsem.

Domyślny `HashingEmbedder` nie wymaga pobierania modelu ani klucza API, dzięki
temu testy i demo działają natychmiast po `git clone`. Do realnego użycia
podstawia się `SentenceTransformerEmbedder` (lokalny model) albo
`OpenAIEmbedder` (API), bez zmiany reszty kodu.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Wspólny interfejs modelu embeddingowego."""

    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Zamienia listę tekstów na macierz wektorów.

        Returns:
            Macierz (len(texts), dimension), wiersze znormalizowane do długości 1.
        """

    def embed_one(self, text: str) -> np.ndarray:
        """Embeduje pojedynczy tekst. Zwraca wektor 1-D."""
        return self.embed([text])[0]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """Normalizuje wiersze do długości 1.

        Po normalizacji iloczyn skalarny równa się podobieństwu kosinusowemu,
        więc wyszukiwanie sprowadza się do jednego mnożenia macierzy.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Zero-wektory zostawiamy jako zera zamiast dzielić przez zero.
        norms[norms == 0] = 1.0
        return matrix / norms


class HashingEmbedder(Embedder):
    """Deterministyczne embeddingi oparte na haszowaniu n-gramów znakowych.

    Nie jest to model semantyczny, nie rozpozna, że "auto" i "samochód" znaczą
    to samo. Łapie natomiast podobieństwo powierzchniowe i formy fleksyjne, co
    dla polszczyzny działa zaskakująco przyzwoicie i wystarcza do
    zademonstrowania architektury oraz do testów.

    Zaleta: zero zależności, zero pobierania, powtarzalny wynik.

    Args:
        dimension: Rozmiar wektora wyjściowego.
        ngram_range: Zakres długości n-gramów znakowych.
    """

    def __init__(
        self, dimension: int = 384, ngram_range: tuple[int, int] = (3, 5)
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension musi być dodatnie")
        self.dimension = dimension
        self.ngram_range = ngram_range

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\sąćęłńóśźż]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _ngrams(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        grams: list[str] = []
        low, high = self.ngram_range

        # N-gramy wyrazowe łapią kolokacje.
        words = text.split()
        grams.extend(words)

        # N-gramy znakowe w obrębie słów łapią fleksję.
        for word in words:
            padded = f" {word} "
            for n in range(low, high + 1):
                for i in range(len(padded) - n + 1):
                    grams.append(padded[i : i + n])

        return grams

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            grams = self._ngrams(text)
            if not grams:
                continue
            for gram in grams:
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimension
                # Znak z piątego bajtu, zmniejsza kolizje kumulujące się w
                # jedną stronę.
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, bucket] += sign

        return self._normalize(matrix)


class SentenceTransformerEmbedder(Embedder):
    """Adapter na sentence-transformers, realne embeddingi semantyczne.

    Wymaga `pip install sentence-transformers`. Przy pierwszym użyciu pobiera
    model (kilkaset MB).

    Dla polszczyzny sprawdzają się modele wielojęzyczne, np.
    "paraphrase-multilingual-MiniLM-L12-v2" (384 wymiary, szybki) albo
    "intfloat/multilingual-e5-base" (768 wymiarów, dokładniejszy).

    Args:
        model_name: Nazwa modelu z Hugging Face.
        batch_size: Rozmiar partii przy kodowaniu.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder wymaga pakietu sentence-transformers. "
                "Zainstaluj: pip install 'rag-demo-pl[st]'"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._normalize(np.asarray(vectors, dtype=np.float32))


class OpenAIEmbedder(Embedder):
    """Adapter na embeddingi OpenAI.

    Wymaga zmiennej środowiskowej OPENAI_API_KEY i pakietu `openai`.

    Args:
        model_name: Model embeddingowy.
        batch_size: Ile tekstów na jedno wywołanie API.
    """

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbedder wymaga pakietu openai. "
                "Zainstaluj: pip install 'rag-demo-pl[openai]'"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = self.DIMENSIONS.get(model_name, 1536)
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model_name, input=batch
            )
            vectors.extend(item.embedding for item in response.data)

        return self._normalize(np.asarray(vectors, dtype=np.float32))


class CachedEmbedder(Embedder):
    """Dekorator cachujący embeddingi w pamięci.

    Ten sam tekst nie jest kodowany dwa razy. Przy powtarzalnych zapytaniach
    (np. w ewaluacji) oszczędza czas i, przy embedderach płatnych, pieniądze.

    Args:
        inner: Embedder do opakowania.
        max_size: Maksymalna liczba zapamiętanych tekstów.
    """

    def __init__(self, inner: Embedder, max_size: int = 10_000) -> None:
        self._inner = inner
        self.dimension = inner.dimension
        self._cache: dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        missing_idx: list[int] = []
        missing_txt: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                result[i] = cached
                self.hits += 1
            else:
                missing_idx.append(i)
                missing_txt.append(text)
                self.misses += 1

        if missing_txt:
            fresh = self._inner.embed(missing_txt)
            for slot, (i, text) in enumerate(
                zip(missing_idx, missing_txt, strict=True)
            ):
                result[i] = fresh[slot]
                if len(self._cache) < self.max_size:
                    self._cache[text] = fresh[slot]

        return result

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
