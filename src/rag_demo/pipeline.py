"""Kompletny pipeline RAG, od dokumentów do odpowiedzi z cytowaniami."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .chunking import Chunk, Chunker, SentenceChunker
from .embeddings import Embedder, HashingEmbedder
from .store import SearchResult, VectorStore

DEFAULT_PROMPT = """Odpowiedz na pytanie wyłącznie na podstawie poniższych fragmentów.
Jeśli fragmenty nie zawierają odpowiedzi, napisz: "Nie znajduję odpowiedzi w dostępnych materiałach."
Powołuj się na źródła, podając numer fragmentu w nawiasach kwadratowych, np. [1].

FRAGMENTY:
{context}

PYTANIE: {question}

ODPOWIEDŹ:"""


class Generator(Protocol):
    """Interfejs generatora odpowiedzi.

    Celowo minimalny, podstawia się tu wywołanie dowolnego LLM-a, funkcję
    testową albo klienta z innego projektu.
    """

    def __call__(self, prompt: str) -> str: ...


@dataclass
class RAGResponse:
    """Odpowiedź systemu wraz z materiałem źródłowym.

    Attributes:
        answer: Wygenerowana odpowiedź. None, gdy generator nie został podany.
        sources: Fragmenty użyte jako kontekst.
        question: Oryginalne pytanie.
        prompt: Pełny prompt wysłany do modelu, kluczowy przy diagnozie.
        metadata: Dodatkowe informacje o przebiegu (tryb, liczba znaków).
    """

    question: str
    sources: list[SearchResult]
    answer: str | None = None
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def citations(self) -> str:
        """Formatuje listę źródeł do wyświetlenia pod odpowiedzią."""
        if not self.sources:
            return "Brak źródeł."

        lines = []
        for i, result in enumerate(self.sources, start=1):
            section = result.chunk.metadata.get("section", "")
            location = f" ({section})" if section else ""
            lines.append(
                f"[{i}] {result.chunk.doc_id}{location}, "
                f"dopasowanie {result.score:.3f}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        parts = [self.answer or "(brak generatora, zwrócono same fragmenty)"]
        parts.append("\nŹRÓDŁA:")
        parts.append(self.citations())
        return "\n".join(parts)


class RAGPipeline:
    """Spina chunking, indeksowanie, wyszukiwanie i generowanie.

    Przykład:
        >>> pipeline = RAGPipeline()
        >>> pipeline.index({"doc1": "treść dokumentu..."})
        >>> odpowiedz = pipeline.query("pytanie")
        >>> print(odpowiedz.citations())

    Args:
        embedder: Model embeddingowy. Domyślnie HashingEmbedder (bez zależności).
        chunker: Strategia dzielenia. Domyślnie SentenceChunker.
        generator: Funkcja generująca odpowiedź. None = zwracane są same fragmenty.
        prompt_template: Szablon promptu z polami {context} i {question}.
        top_k: Ile fragmentów trafia do kontekstu.
        search_mode: "dense", "lexical" albo "hybrid".
        alpha: Waga wyszukiwania gęstego w trybie hybrydowym.
        max_context_chars: Twardy limit długości kontekstu.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        chunker: Chunker | None = None,
        generator: Generator | None = None,
        prompt_template: str = DEFAULT_PROMPT,
        top_k: int = 5,
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        max_context_chars: int = 8000,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunker = chunker or SentenceChunker()
        self.generator = generator
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.search_mode = search_mode
        self.alpha = alpha
        self.max_context_chars = max_context_chars
        self.store = VectorStore(self.embedder)

    def index(self, documents: dict[str, str], **metadata: Any) -> int:
        """Dzieli dokumenty i dodaje je do indeksu.

        Args:
            documents: Mapa doc_id → treść.
            **metadata: Metadane dołączane do wszystkich fragmentów.

        Returns:
            Liczba dodanych fragmentów.
        """
        chunks = self.chunker.split_many(documents, **metadata)
        self.store.add(chunks)
        return len(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Dodaje gotowe fragmenty, pomijając etap chunkingu."""
        self.store.add(chunks)
        return len(chunks)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """Zwraca fragmenty najlepiej pasujące do pytania."""
        return self.store.search(
            question,
            top_k=top_k or self.top_k,
            mode=self.search_mode,
            alpha=self.alpha,
        )

    def build_context(self, results: list[SearchResult]) -> str:
        """Skleja fragmenty w kontekst, respektując limit długości.

        Fragmenty numerowane są od 1, żeby model mógł się na nie powoływać
        w formacie [1], [2], i żeby dało się to potem zweryfikować.
        """
        parts: list[str] = []
        total = 0

        for i, result in enumerate(results, start=1):
            section = result.chunk.metadata.get("section", "")
            header = f"[{i}] {result.chunk.doc_id}"
            if section:
                header += f", {section}"

            block = f"{header}\n{result.chunk.text}"

            if total + len(block) > self.max_context_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        generator: Generator | None = None,
    ) -> RAGResponse:
        """Wykonuje pełny cykl: wyszukanie → kontekst → generowanie.

        Args:
            question: Pytanie użytkownika.
            top_k: Nadpisuje domyślną liczbę fragmentów.
            generator: Nadpisuje generator ustawiony w konstruktorze.

        Returns:
            RAGResponse z odpowiedzią, źródłami i użytym promptem.
        """
        results = self.retrieve(question, top_k=top_k)
        context = self.build_context(results)
        prompt = self.prompt_template.format(context=context, question=question)

        gen = generator or self.generator
        answer = gen(prompt) if gen is not None else None

        return RAGResponse(
            question=question,
            sources=results,
            answer=answer,
            prompt=prompt,
            metadata={
                "search_mode": self.search_mode,
                "alpha": self.alpha,
                "chunks_retrieved": len(results),
                "context_chars": len(context),
                "chunks_in_context": context.count("\n\n") + 1 if context else 0,
            },
        )

    def stats(self) -> dict[str, Any]:
        """Statystyki indeksu."""
        return self.store.stats()
