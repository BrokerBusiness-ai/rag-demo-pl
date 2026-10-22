"""Strategie dzielenia tekstu na fragmenty.

Chunking decyduje o jakości całego RAG-a bardziej niż wybór modelu embeddingowego.
Zbyt duże fragmenty rozmywają sygnał, zbyt małe gubią kontekst. Ten moduł daje
trzy strategie o rosnącej świadomości struktury tekstu.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """Fragment tekstu wraz z metadanymi pochodzenia.

    Attributes:
        text: Treść fragmentu.
        doc_id: Identyfikator dokumentu źródłowego.
        index: Numer fragmentu w obrębie dokumentu (od 0).
        start: Pozycja początkowa w oryginalnym tekście.
        end: Pozycja końcowa w oryginalnym tekście.
        metadata: Dowolne dodatkowe dane (tytuł, sekcja, URL).
    """

    text: str
    doc_id: str
    index: int
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __str__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        suffix = "..." if len(self.text) > 60 else ""
        return f"[{self.doc_id}#{self.index}] {preview}{suffix}"


class Chunker(ABC):
    """Wspólny interfejs strategii chunkingu."""

    @abstractmethod
    def split(self, text: str, doc_id: str, **metadata: Any) -> list[Chunk]:
        """Dzieli tekst na fragmenty."""

    def split_many(
        self, documents: dict[str, str], **metadata: Any
    ) -> list[Chunk]:
        """Dzieli wiele dokumentów naraz.

        Args:
            documents: Mapa doc_id → treść.
            **metadata: Metadane dołączane do każdego fragmentu.
        """
        chunks: list[Chunk] = []
        for doc_id, text in documents.items():
            chunks.extend(self.split(text, doc_id, **metadata))
        return chunks


class FixedSizeChunker(Chunker):
    """Dzieli na fragmenty o stałej długości znakowej, z zakładką.

    Najprostsza strategia. Szybka i przewidywalna, ale tnie w środku zdań -
    fragment może zaczynać się od połowy słowa. Użyteczna jako punkt odniesienia
    przy porównywaniu jakości innych strategii.

    Args:
        chunk_size: Docelowa długość fragmentu w znakach.
        overlap: Ile znaków końcówki poprzedniego fragmentu powtórzyć na początku
            następnego. Zakładka ratuje zdania przecięte na granicy.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size musi być dodatnie")
        if overlap < 0:
            raise ValueError("overlap nie może być ujemny")
        if overlap >= chunk_size:
            raise ValueError("overlap musi być mniejszy niż chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, doc_id: str, **metadata: Any) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        position = 0
        index = 0

        while position < len(text):
            end = min(position + self.chunk_size, len(text))
            fragment = text[position:end].strip()

            if fragment:
                chunks.append(
                    Chunk(
                        text=fragment,
                        doc_id=doc_id,
                        index=index,
                        start=position,
                        end=end,
                        metadata=dict(metadata),
                    )
                )
                index += 1

            if end >= len(text):
                break
            position += step

        return chunks


class SentenceChunker(Chunker):
    """Grupuje całe zdania, nie przekraczając limitu długości.

    Nigdy nie tnie w środku zdania, więc fragmenty są czytelne i lepiej się
    embedują. To rozsądny domyślny wybór dla większości tekstów.

    Segmentacja zdań uwzględnia polskie skróty (np., itd., dr, prof.), które
    naiwny podział po kropce błędnie traktuje jako koniec zdania.

    Args:
        max_chars: Górny limit długości fragmentu.
        overlap_sentences: Ile ostatnich zdań powtórzyć w kolejnym fragmencie.
    """

    # Skróty, po których kropka nie kończy zdania.
    ABBREVIATIONS = {
        "np", "itd", "itp", "tj", "tzn", "tzw", "m.in", "ok", "por", "zob",
        "ur", "zm", "wg", "ww", "ds", "dr", "hab", "prof", "inż", "mgr",
        "św", "ul", "al", "pl", "godz", "min", "sek", "str", "nr", "poz",
    }

    def __init__(
        self, max_chars: int = 800, overlap_sentences: int = 1
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars musi być dodatnie")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences nie może być ujemny")

        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        """Dzieli tekst na zdania, respektując polskie skróty."""
        # Kandydaci na granicę: kropka/wykrzyknik/pytajnik + spacja + wielka litera.
        pattern = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])")
        raw_parts = pattern.split(text)

        sentences: list[str] = []
        buffer = ""

        for part in raw_parts:
            candidate = f"{buffer} {part}".strip() if buffer else part
            # Sprawdzamy, czy fragment kończy się skrótem, wtedy kropka nie
            # była końcem zdania i doklejamy następny fragment.
            trimmed = candidate.rstrip(".!?").split()
            last_word = trimmed[-1] if trimmed else ""
            if last_word.lower().rstrip(".") in self.ABBREVIATIONS:
                buffer = candidate
                continue
            sentences.append(candidate)
            buffer = ""

        if buffer:
            sentences.append(buffer)

        return [s.strip() for s in sentences if s.strip()]

    def split(self, text: str, doc_id: str, **metadata: Any) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        index = 0
        cursor = 0

        def flush() -> None:
            nonlocal index, cursor
            if not current:
                return
            fragment = " ".join(current)
            start = text.find(current[0], cursor)
            if start == -1:
                start = cursor
            end = start + len(fragment)
            chunks.append(
                Chunk(
                    text=fragment,
                    doc_id=doc_id,
                    index=index,
                    start=start,
                    end=end,
                    metadata=dict(metadata),
                )
            )
            index += 1
            cursor = start + len(current[0])

        for sentence in sentences:
            sentence_len = len(sentence)

            # Zdanie dłuższe niż limit trafia do własnego fragmentu.
            if sentence_len > self.max_chars:
                flush()
                current, current_len = [], 0
                chunks.append(
                    Chunk(
                        text=sentence,
                        doc_id=doc_id,
                        index=index,
                        start=cursor,
                        end=cursor + sentence_len,
                        metadata=dict(metadata),
                    )
                )
                index += 1
                cursor += sentence_len
                continue

            if current_len + sentence_len > self.max_chars and current:
                flush()
                tail = (
                    current[-self.overlap_sentences :]
                    if self.overlap_sentences
                    else []
                )
                current = [*tail, sentence]
                current_len = sum(len(s) for s in current)
            else:
                current.append(sentence)
                current_len += sentence_len

        flush()
        return chunks


class MarkdownChunker(Chunker):
    """Dzieli po nagłówkach Markdown, zachowując hierarchię sekcji.

    Najlepsza strategia dla dokumentacji i artykułów. Każdy fragment niesie
    ścieżkę nagłówków (np. "Instalacja > Wymagania"), co daje modelowi kontekst
    o tym, skąd fragment pochodzi, i pozwala go pokazać w cytowaniu.

    Sekcje dłuższe niż max_chars są dodatkowo dzielone zdaniowo.

    Args:
        max_chars: Limit długości fragmentu.
        min_chars: Sekcje krótsze niż to są scalane z następną.
    """

    HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chars: int = 1000, min_chars: int = 100) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self._fallback = SentenceChunker(max_chars=max_chars)

    def _sections(self, text: str) -> Iterator[tuple[str, str, int]]:
        """Zwraca (ścieżka_nagłówków, treść, pozycja_startowa)."""
        matches = list(self.HEADING.finditer(text))

        if not matches:
            yield ("", text, 0)
            return

        # Tekst przed pierwszym nagłówkiem.
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                yield ("", preamble, 0)

        stack: list[tuple[int, str]] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            body_start = match.end()
            body_end = (
                matches[i + 1].start() if i + 1 < len(matches) else len(text)
            )
            body = text[body_start:body_end].strip()

            path = " > ".join(t for _, t in stack)
            if body:
                yield (path, body, body_start)

    def split(self, text: str, doc_id: str, **metadata: Any) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        index = 0
        pending: tuple[str, str, int] | None = None

        for path, body, start in self._sections(text):
            # Scalanie zbyt krótkich sekcji z następną.
            if pending is not None:
                prev_path, prev_body, prev_start = pending
                body = f"{prev_body}\n\n{body}"
                path = prev_path or path
                start = prev_start
                pending = None

            if len(body) < self.min_chars:
                pending = (path, body, start)
                continue

            section_meta = dict(metadata)
            if path:
                section_meta["section"] = path

            if len(body) <= self.max_chars:
                chunks.append(
                    Chunk(
                        text=body,
                        doc_id=doc_id,
                        index=index,
                        start=start,
                        end=start + len(body),
                        metadata=section_meta,
                    )
                )
                index += 1
            else:
                # Długa sekcja, tniemy zdaniowo, zachowując metadane sekcji.
                for sub in self._fallback.split(body, doc_id, **section_meta):
                    chunks.append(
                        Chunk(
                            text=sub.text,
                            doc_id=doc_id,
                            index=index,
                            start=start + sub.start,
                            end=start + sub.end,
                            metadata=section_meta,
                        )
                    )
                    index += 1

        # Ostatnia zbyt krótka sekcja i tak trafia do wyniku.
        if pending is not None:
            path, body, start = pending
            section_meta = dict(metadata)
            if path:
                section_meta["section"] = path
            chunks.append(
                Chunk(
                    text=body,
                    doc_id=doc_id,
                    index=index,
                    start=start,
                    end=start + len(body),
                    metadata=section_meta,
                )
            )

        return chunks
