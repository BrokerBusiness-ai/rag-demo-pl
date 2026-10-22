"""Testy magazynu wektorów i wyszukiwania."""

import numpy as np
import pytest

from rag_demo.chunking import Chunk, SentenceChunker
from rag_demo.embeddings import HashingEmbedder
from rag_demo.store import BM25, VectorStore


@pytest.fixture
def embedder():
    return HashingEmbedder(dimension=128)


@pytest.fixture
def chunks():
    teksty = {
        "python": "Python to język programowania wysokiego poziomu. "
                  "Jest popularny w analizie danych i uczeniu maszynowym.",
        "rag": "RAG łączy wyszukiwanie dokumentów z generowaniem odpowiedzi. "
               "Dzięki temu model odpowiada na podstawie źródeł.",
        "embedding": "Embedding to reprezentacja tekstu w postaci wektora liczb. "
                     "Podobne teksty mają podobne wektory.",
    }
    return SentenceChunker(max_chars=500).split_many(teksty)


@pytest.fixture
def store(embedder, chunks):
    s = VectorStore(embedder)
    s.add(chunks)
    return s


class TestBM25:
    def test_znajduje_dokument_z_terminem(self):
        bm25 = BM25()
        bm25.fit(["kot siedzi na macie", "pies biega po ogrodzie"])
        scores = bm25.scores("kot")

        assert scores[0] > scores[1]

    def test_brak_dopasowania_daje_zera(self):
        bm25 = BM25()
        bm25.fit(["kot", "pies"])
        scores = bm25.scores("samolot")

        assert np.allclose(scores, 0.0)

    def test_pusty_korpus(self):
        assert BM25().scores("cokolwiek").size == 0

    def test_tokenizacja_polskich_znakow(self):
        tokeny = BM25.tokenize("Zażółć gęślą jaźń")
        assert "zażółć" in tokeny
        assert "gęślą" in tokeny


class TestVectorStore:
    def test_dodaje_fragmenty(self, store, chunks):
        assert len(store) == len(chunks)

    def test_pusty_magazyn_zwraca_pusta_liste(self, embedder):
        assert VectorStore(embedder).search("pytanie") == []

    def test_dodanie_pustej_listy_nic_nie_zmienia(self, store):
        przed = len(store)
        store.add([])
        assert len(store) == przed

    def test_wyszukiwanie_zwraca_wyniki(self, store):
        wyniki = store.search("Python programowanie", top_k=2)
        assert len(wyniki) <= 2
        assert all(w.score >= 0 for w in wyniki)

    def test_wyniki_posortowane_malejaco(self, store):
        wyniki = store.search("wektor embedding", top_k=3)
        oceny = [w.score for w in wyniki]
        assert oceny == sorted(oceny, reverse=True)

    def test_rangi_numerowane_od_jeden(self, store):
        wyniki = store.search("Python", top_k=3)
        assert [w.rank for w in wyniki] == list(range(1, len(wyniki) + 1))

    def test_top_k_ogranicza_liczbe_wynikow(self, store):
        assert len(store.search("tekst", top_k=1)) == 1

    def test_tryb_dense(self, store):
        wyniki = store.search("uczenie maszynowe", mode="dense", top_k=2)
        assert len(wyniki) > 0

    def test_tryb_lexical(self, store):
        wyniki = store.search("Python", mode="lexical", top_k=2)
        assert len(wyniki) > 0

    def test_tryb_hybrid(self, store):
        wyniki = store.search("Python", mode="hybrid", top_k=2)
        assert len(wyniki) > 0

    def test_nieznany_tryb_rzuca_blad(self, store):
        with pytest.raises(ValueError, match="Nieznany tryb"):
            store.search("x", mode="magiczny")

    def test_lexical_bez_bm25_rzuca_blad(self, embedder, chunks):
        s = VectorStore(embedder, enable_bm25=False)
        s.add(chunks)
        with pytest.raises(RuntimeError, match="leksykalne"):
            s.search("x", mode="lexical")

    def test_bm25_znajduje_rzadki_termin_lepiej_niz_dense(self, embedder):
        """BM25 ma przewagę przy dosłownych dopasowaniach rzadkich terminów."""
        chunks = [
            Chunk("Identyfikator zamówienia to XK-9931-QQ.", "faktura", 0, 0, 40),
            Chunk("Ogólne informacje o procesie zamawiania towaru.", "info", 0, 0, 46),
        ]
        store = VectorStore(embedder)
        store.add(chunks)

        wyniki = store.search("XK-9931-QQ", mode="lexical", top_k=1)
        assert wyniki[0].chunk.doc_id == "faktura"

    def test_min_score_odcina_slabe_wyniki(self, store):
        wszystkie = store.search("tekst", top_k=10, min_score=0.0)
        odfiltrowane = store.search("tekst", top_k=10, min_score=0.99)
        assert len(odfiltrowane) <= len(wszystkie)


class TestPersistence:
    def test_zapis_i_odczyt(self, store, embedder, tmp_path):
        sciezka = tmp_path / "index.pkl"
        store.save(sciezka)

        wczytany = VectorStore.load(sciezka, embedder)
        assert len(wczytany) == len(store)

    def test_wyniki_identyczne_po_wczytaniu(self, store, embedder, tmp_path):
        sciezka = tmp_path / "index.pkl"
        store.save(sciezka)
        wczytany = VectorStore.load(sciezka, embedder)

        a = store.search("Python", top_k=2)
        b = wczytany.search("Python", top_k=2)

        assert [r.chunk.text for r in a] == [r.chunk.text for r in b]

    def test_niezgodny_wymiar_rzuca_blad(self, store, tmp_path):
        sciezka = tmp_path / "index.pkl"
        store.save(sciezka)

        inny = HashingEmbedder(dimension=999)
        with pytest.raises(ValueError, match="Niezgodność wymiarów"):
            VectorStore.load(sciezka, inny)

    def test_tworzy_katalog_docelowy(self, store, tmp_path):
        sciezka = tmp_path / "gleboko" / "index.pkl"
        store.save(sciezka)
        assert sciezka.exists()


class TestStats:
    def test_pusty_magazyn(self, embedder):
        assert VectorStore(embedder).stats() == {"chunks": 0}

    def test_statystyki_zawieraja_kluczowe_pola(self, store):
        stats = store.stats()
        assert stats["chunks"] > 0
        assert stats["documents"] == 3
        assert "avg_chunk_chars" in stats
        assert stats["bm25_enabled"] is True
