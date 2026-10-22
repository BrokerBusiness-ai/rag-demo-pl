"""Metryki jakości retrievalu.

Bez pomiaru RAG jest zgadywanką, "wydaje się działać" to nie jest odpowiedź na
pytanie, czy zmiana chunkingu pomogła, czy zaszkodziła. Ten moduł liczy metryki,
które pozwalają porównywać konfiguracje liczbowo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .store import VectorStore


@dataclass(frozen=True)
class EvalCase:
    """Pojedynczy przypadek testowy.

    Attributes:
        query: Zapytanie.
        relevant_doc_ids: Identyfikatory dokumentów uznanych za trafne.
        note: Opcjonalny komentarz, po co ten przypadek istnieje.
    """

    query: str
    relevant_doc_ids: set[str]
    note: str = ""


@dataclass
class EvalResult:
    """Wyniki ewaluacji dla zestawu przypadków.

    Attributes:
        recall_at_k: Odsetek przypadków, w których trafny dokument znalazł się
            w top-k. Najważniejsza metryka dla RAG, jeśli dokumentu nie ma
            w kontekście, model nie ma szans odpowiedzieć poprawnie.
        precision_at_k: Jaka część zwróconych wyników była trafna.
        mrr: Mean Reciprocal Rank, nagradza wysokie pozycje trafnego wyniku.
        hit_rate: Odsetek zapytań z choć jednym trafieniem.
        per_case: Szczegóły dla każdego przypadku.
    """

    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate: float
    k: int
    mode: str
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Tryb: {self.mode}, k={self.k}\n"
            f"  Recall@{self.k}:    {self.recall_at_k:.3f}\n"
            f"  Precision@{self.k}: {self.precision_at_k:.3f}\n"
            f"  MRR:              {self.mrr:.3f}\n"
            f"  Hit rate:         {self.hit_rate:.3f}"
        )

    def failures(self) -> list[dict[str, Any]]:
        """Przypadki bez trafienia, od nich zaczyna się poprawianie systemu."""
        return [case for case in self.per_case if not case["hit"]]


def evaluate(
    store: VectorStore,
    cases: list[EvalCase],
    k: int = 5,
    mode: str = "hybrid",
    alpha: float = 0.5,
) -> EvalResult:
    """Liczy metryki retrievalu dla zestawu przypadków.

    Args:
        store: Zaindeksowany magazyn.
        cases: Przypadki testowe.
        k: Ile wyników brać pod uwagę.
        mode: Tryb wyszukiwania.
        alpha: Waga wyszukiwania gęstego w trybie hybrydowym.

    Returns:
        Zagregowane metryki plus szczegóły per przypadek.
    """
    if not cases:
        return EvalResult(0.0, 0.0, 0.0, 0.0, k, mode)

    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    hits = 0
    details: list[dict[str, Any]] = []

    for case in cases:
        results = store.search(case.query, top_k=k, mode=mode, alpha=alpha)
        retrieved_ids = [r.chunk.doc_id for r in results]
        relevant_found = set(retrieved_ids) & case.relevant_doc_ids

        recall = (
            len(relevant_found) / len(case.relevant_doc_ids)
            if case.relevant_doc_ids
            else 0.0
        )
        precision = (
            sum(1 for d in retrieved_ids if d in case.relevant_doc_ids)
            / len(retrieved_ids)
            if retrieved_ids
            else 0.0
        )

        rr = 0.0
        for position, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in case.relevant_doc_ids:
                rr = 1.0 / position
                break

        hit = bool(relevant_found)
        hits += int(hit)

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(rr)

        details.append(
            {
                "query": case.query,
                "expected": sorted(case.relevant_doc_ids),
                "retrieved": retrieved_ids,
                "hit": hit,
                "recall": round(recall, 3),
                "reciprocal_rank": round(rr, 3),
                "top_score": round(results[0].score, 3) if results else 0.0,
                "note": case.note,
            }
        )

    n = len(cases)
    return EvalResult(
        recall_at_k=sum(recalls) / n,
        precision_at_k=sum(precisions) / n,
        mrr=sum(reciprocal_ranks) / n,
        hit_rate=hits / n,
        k=k,
        mode=mode,
        per_case=details,
    )


def compare_modes(
    store: VectorStore,
    cases: list[EvalCase],
    k: int = 5,
    alphas: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0),
) -> list[EvalResult]:
    """Porównuje tryby wyszukiwania i wagi hybrydy.

    Odpowiada na pytanie, które w praktyce pada najczęściej: czy w tym korpusie
    lepiej działają embeddingi, BM25, czy jakieś ich połączenie, i w jakich
    proporcjach.

    Returns:
        Wyniki posortowane malejąco po recall@k.
    """
    results = [
        evaluate(store, cases, k=k, mode="dense"),
        evaluate(store, cases, k=k, mode="lexical"),
    ]

    for alpha in alphas:
        result = evaluate(store, cases, k=k, mode="hybrid", alpha=alpha)
        result.mode = f"hybrid(α={alpha})"
        results.append(result)

    return sorted(results, key=lambda r: r.recall_at_k, reverse=True)


def format_comparison(results: list[EvalResult]) -> str:
    """Formatuje porównanie trybów jako tabelę tekstową."""
    header = f"{'Tryb':<18} {'Recall':>8} {'Precision':>10} {'MRR':>8} {'Hit':>8}"
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r.mode:<18} {r.recall_at_k:>8.3f} {r.precision_at_k:>10.3f} "
            f"{r.mrr:>8.3f} {r.hit_rate:>8.3f}"
        )

    return "\n".join(lines)
