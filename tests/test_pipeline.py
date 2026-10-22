"""Testy pipeline'u RAG i ewaluacji."""

import pytest

from rag_demo.chunking import MarkdownChunker
from rag_demo.embeddings import CachedEmbedder, HashingEmbedder
from rag_demo.evaluation import EvalCase, compare_modes, evaluate, format_comparison
from rag_demo.pipeline import RAGPipeline


@pytest.fixture
def dokumenty():
    return {
        "python": (
            "Python to język programowania stworzony przez Guido van Rossuma. "
            "Jest znany z czytelnej składni i szerokiego ekosystemu bibliotek. "
            "W analizie danych używa się pakietów pandas oraz numpy."
        ),
        "rag": (
            "RAG to skrót od Retrieval Augmented Generation. "
            "System najpierw wyszukuje fragmenty dokumentów pasujące do pytania. "
            "Następnie przekazuje je modelowi językowemu jako kontekst."
        ),
        "wektory": (
            "Baza wektorowa przechowuje reprezentacje tekstu w postaci liczb. "
            "Wyszukiwanie polega na znajdowaniu najbliższych sąsiadów. "
            "Popularne rozwiązania to Pinecone, Weaviate oraz pgvector."
        ),
    }


@pytest.fixture
def pipeline(dokumenty):
    p = RAGPipeline(embedder=HashingEmbedder(dimension=256))
    p.index(dokumenty)
    return p


class TestIndexing:
    def test_indeksowanie_zwraca_liczbe_fragmentow(self, dokumenty):
        p = RAGPipeline()
        liczba = p.index(dokumenty)
        assert liczba > 0
        assert len(p.store) == liczba

    def test_statystyki_po_indeksowaniu(self, pipeline):
        stats = pipeline.stats()
        assert stats["documents"] == 3
        assert stats["chunks"] > 0

    def test_indeksowanie_z_metadanymi(self, dokumenty):
        p = RAGPipeline()
        p.index(dokumenty, zrodlo="test")
        assert all(c.metadata.get("zrodlo") == "test" for c in p.store.chunks)


class TestRetrieval:
    def test_zwraca_fragmenty(self, pipeline):
        wyniki = pipeline.retrieve("Co to jest RAG?")
        assert len(wyniki) > 0

    def test_znajduje_wlasciwy_dokument(self, pipeline):
        wyniki = pipeline.retrieve("pandas numpy analiza danych", top_k=3)
        assert "python" in [r.chunk.doc_id for r in wyniki]

    def test_top_k_ogranicza_wyniki(self, pipeline):
        assert len(pipeline.retrieve("tekst", top_k=2)) <= 2


class TestContextBuilding:
    def test_kontekst_zawiera_numerowane_fragmenty(self, pipeline):
        wyniki = pipeline.retrieve("RAG", top_k=2)
        kontekst = pipeline.build_context(wyniki)

        assert "[1]" in kontekst
        assert "[2]" in kontekst

    def test_respektuje_limit_dlugosci(self, dokumenty):
        p = RAGPipeline(max_context_chars=100)
        p.index(dokumenty)
        wyniki = p.retrieve("tekst", top_k=10)
        kontekst = p.build_context(wyniki)

        assert len(kontekst) <= 200  # limit + jeden blok tolerancji

    def test_pusty_wynik_daje_pusty_kontekst(self, pipeline):
        assert pipeline.build_context([]) == ""


class TestQuery:
    def test_bez_generatora_zwraca_same_zrodla(self, pipeline):
        odpowiedz = pipeline.query("Co to jest RAG?")

        assert odpowiedz.answer is None
        assert len(odpowiedz.sources) > 0
        assert odpowiedz.prompt != ""

    def test_z_generatorem_zwraca_odpowiedz(self, pipeline):
        def fake_generator(prompt: str) -> str:
            return "To jest odpowiedź testowa."

        odpowiedz = pipeline.query("Co to jest RAG?", generator=fake_generator)
        assert odpowiedz.answer == "To jest odpowiedź testowa."

    def test_generator_dostaje_kontekst_w_promptcie(self, pipeline):
        przechwycone = {}

        def generator(prompt: str) -> str:
            przechwycone["prompt"] = prompt
            return "ok"

        pipeline.query("RAG", generator=generator)

        assert "FRAGMENTY:" in przechwycone["prompt"]
        assert "PYTANIE: RAG" in przechwycone["prompt"]

    def test_cytowania_formatowane(self, pipeline):
        odpowiedz = pipeline.query("RAG")
        cytowania = odpowiedz.citations()

        assert "[1]" in cytowania

    def test_metadane_opisuja_przebieg(self, pipeline):
        odpowiedz = pipeline.query("RAG", top_k=2)

        assert odpowiedz.metadata["chunks_retrieved"] <= 2
        assert odpowiedz.metadata["search_mode"] == "hybrid"
        assert odpowiedz.metadata["context_chars"] > 0


class TestEvaluation:
    def test_ewaluacja_liczy_metryki(self, pipeline):
        przypadki = [
            EvalCase("pandas numpy dane", {"python"}),
            EvalCase("retrieval augmented generation", {"rag"}),
            EvalCase("Pinecone Weaviate pgvector", {"wektory"}),
        ]

        wynik = evaluate(pipeline.store, przypadki, k=3)

        assert 0.0 <= wynik.recall_at_k <= 1.0
        assert 0.0 <= wynik.mrr <= 1.0
        assert len(wynik.per_case) == 3

    def test_pusta_lista_przypadkow(self, pipeline):
        wynik = evaluate(pipeline.store, [], k=5)
        assert wynik.recall_at_k == 0.0

    def test_summary_zawiera_metryki(self, pipeline):
        przypadki = [EvalCase("RAG", {"rag"})]
        summary = evaluate(pipeline.store, przypadki).summary()

        assert "Recall" in summary
        assert "MRR" in summary

    def test_failures_zwraca_nietrafione(self, pipeline):
        przypadki = [
            EvalCase("pandas numpy", {"python"}),
            EvalCase("zupełnie niezwiązane pytanie xyz", {"nieistniejacy"}),
        ]
        wynik = evaluate(pipeline.store, przypadki, k=1)

        assert len(wynik.failures()) >= 1

    def test_porownanie_trybow(self, pipeline):
        przypadki = [
            EvalCase("pandas numpy dane", {"python"}),
            EvalCase("Pinecone Weaviate", {"wektory"}),
        ]
        wyniki = compare_modes(pipeline.store, przypadki, k=3, alphas=(0.0, 1.0))

        assert len(wyniki) == 4  # dense, lexical, 2x hybrid
        # Posortowane malejąco po recall.
        recalls = [w.recall_at_k for w in wyniki]
        assert recalls == sorted(recalls, reverse=True)

    def test_formatowanie_porownania(self, pipeline):
        przypadki = [EvalCase("RAG", {"rag"})]
        wyniki = compare_modes(pipeline.store, przypadki, k=2, alphas=(0.5,))
        tabela = format_comparison(wyniki)

        assert "Recall" in tabela
        assert "dense" in tabela


class TestCachedEmbedder:
    def test_drugie_wywolanie_z_cache(self):
        inner = HashingEmbedder(dimension=64)
        cached = CachedEmbedder(inner)

        cached.embed(["tekst"])
        cached.embed(["tekst"])

        assert cached.hits == 1
        assert cached.misses == 1
        assert cached.hit_rate == 0.5

    def test_wyniki_identyczne_z_cache(self):
        cached = CachedEmbedder(HashingEmbedder(dimension=64))

        a = cached.embed(["ten sam tekst"])
        b = cached.embed(["ten sam tekst"])

        assert (a == b).all()

    def test_zachowuje_wymiar_wewnetrznego(self):
        cached = CachedEmbedder(HashingEmbedder(dimension=128))
        assert cached.dimension == 128


class TestMarkdownIntegration:
    def test_pipeline_z_markdown_chunkerem(self):
        md = (
            "# Wprowadzenie\n\nOgólny opis systemu i jego przeznaczenia w praktyce.\n\n"
            "## Instalacja\n\nWymagany jest Python w wersji 3.10 lub nowszej."
        )
        p = RAGPipeline(chunker=MarkdownChunker(max_chars=500, min_chars=10))
        p.index({"dokumentacja": md})

        wyniki = p.retrieve("instalacja Python wersja")
        assert len(wyniki) > 0
        assert any("section" in r.chunk.metadata for r in wyniki)
