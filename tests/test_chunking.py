"""Testy strategii chunkingu."""

import pytest

from rag_demo.chunking import (
    Chunk,
    FixedSizeChunker,
    MarkdownChunker,
    SentenceChunker,
)


@pytest.fixture
def tekst_pl():
    return (
        "Sztuczna inteligencja rozwija się szybko. "
        "Modele językowe potrafią generować tekst. "
        "Systemy RAG łączą wyszukiwanie z generowaniem. "
        "Dzięki temu odpowiedzi są oparte na źródłach."
    )


class TestChunk:
    def test_liczy_znaki_i_slowa(self):
        chunk = Chunk("Ala ma kota", "doc1", 0, 0, 11)
        assert chunk.char_count == 11
        assert chunk.word_count == 3

    def test_str_skraca_dlugi_tekst(self):
        chunk = Chunk("x" * 200, "doc1", 0, 0, 200)
        assert "..." in str(chunk)
        assert len(str(chunk)) < 100


class TestFixedSizeChunker:
    def test_dzieli_na_fragmenty_o_zadanym_rozmiarze(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        chunks = chunker.split("x" * 200, "doc1")

        assert len(chunks) == 4
        assert all(c.char_count <= 50 for c in chunks)

    def test_zakladka_powtarza_koncowke(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = chunker.split("abcdefghij" * 20, "doc1")

        assert len(chunks) > 1
        # Przy zakładce fragmentów jest więcej niż bez niej.
        bez = FixedSizeChunker(chunk_size=50, overlap=0).split(
            "abcdefghij" * 20, "doc1"
        )
        assert len(chunks) > len(bez)

    def test_pusty_tekst_daje_pusta_liste(self):
        assert FixedSizeChunker().split("", "doc1") == []
        assert FixedSizeChunker().split("   \n  ", "doc1") == []

    def test_numeruje_fragmenty_od_zera(self):
        chunks = FixedSizeChunker(chunk_size=30, overlap=0).split("x" * 100, "d")
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_zachowuje_doc_id(self):
        chunks = FixedSizeChunker(chunk_size=30, overlap=5).split(
            "x" * 100, "moj-dok"
        )
        assert all(c.doc_id == "moj-dok" for c in chunks)

    def test_przekazuje_metadane(self):
        chunks = FixedSizeChunker(chunk_size=30, overlap=5).split(
            "x" * 100, "d", zrodlo="wikipedia"
        )
        assert all(c.metadata["zrodlo"] == "wikipedia" for c in chunks)

    def test_walidacja_parametrow(self):
        with pytest.raises(ValueError, match="chunk_size"):
            FixedSizeChunker(chunk_size=0)
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=100, overlap=-1)
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=100, overlap=100)


class TestSentenceChunker:
    def test_nie_tnie_w_srodku_zdania(self, tekst_pl):
        chunks = SentenceChunker(max_chars=100, overlap_sentences=0).split(
            tekst_pl, "doc1"
        )
        for chunk in chunks:
            assert chunk.text.rstrip()[-1] in ".!?"

    def test_respektuje_limit_dlugosci(self, tekst_pl):
        chunks = SentenceChunker(max_chars=80, overlap_sentences=0).split(
            tekst_pl, "doc1"
        )
        # Limit może być przekroczony tylko przez pojedyncze długie zdanie.
        for chunk in chunks:
            assert chunk.char_count <= 80 or len(chunk.text.split(".")) <= 2

    def test_polskie_skroty_nie_koncza_zdania(self):
        tekst = "Używamy narzędzi np. Pythona oraz bibliotek. To jest drugie zdanie."
        chunker = SentenceChunker(max_chars=1000)
        chunks = chunker.split(tekst, "doc1")

        # Całość mieści się w limicie, więc powinien być jeden fragment.
        assert len(chunks) == 1
        assert "np. Pythona" in chunks[0].text

    def test_dr_i_prof_nie_koncza_zdania(self):
        tekst = "Badanie prowadził dr Kowalski. Wyniki opublikowano w 2024 roku."
        chunks = SentenceChunker(max_chars=1000).split(tekst, "doc1")
        assert len(chunks) == 1

    def test_bardzo_dlugie_zdanie_trafia_do_wlasnego_fragmentu(self):
        dlugie = "Słowo " * 200 + "."
        chunks = SentenceChunker(max_chars=100).split(dlugie, "doc1")
        assert len(chunks) >= 1

    def test_pusty_tekst(self):
        assert SentenceChunker().split("", "doc1") == []

    def test_zakladka_zdaniowa(self, tekst_pl):
        z_zakladka = SentenceChunker(max_chars=90, overlap_sentences=1).split(
            tekst_pl, "doc1"
        )
        bez_zakladki = SentenceChunker(max_chars=90, overlap_sentences=0).split(
            tekst_pl, "doc1"
        )
        assert len(z_zakladka) >= len(bez_zakladki)


class TestMarkdownChunker:
    def test_dzieli_po_naglowkach(self):
        md = (
            "# Tytuł\n\nWstęp do dokumentu o odpowiedniej długości tekstu.\n\n"
            "## Sekcja A\n\nTreść sekcji A, wystarczająco długa by nie scalać.\n\n"
            "## Sekcja B\n\nTreść sekcji B, również odpowiednio rozbudowana."
        )
        chunks = MarkdownChunker(max_chars=1000, min_chars=10).split(md, "doc1")
        assert len(chunks) >= 2

    def test_zachowuje_sciezke_naglowkow(self):
        md = (
            "# Instalacja\n\nOgólne informacje o instalacji pakietu w systemie.\n\n"
            "## Wymagania\n\nPotrzebny jest Python w wersji 3.10 lub nowszej."
        )
        chunks = MarkdownChunker(max_chars=1000, min_chars=10).split(md, "doc1")
        sekcje = [c.metadata.get("section", "") for c in chunks]
        assert any("Instalacja" in s for s in sekcje)
        assert any("Wymagania" in s for s in sekcje)

    def test_hierarchia_w_sciezce(self):
        md = (
            "# Główny\n\nTreść głównej sekcji o wystarczającej długości tutaj.\n\n"
            "## Podsekcja\n\nTreść podsekcji, także odpowiednio długa i sensowna."
        )
        chunks = MarkdownChunker(max_chars=1000, min_chars=10).split(md, "doc1")
        sciezki = [c.metadata.get("section", "") for c in chunks]
        assert any(">" in s for s in sciezki)

    def test_tekst_bez_naglowkow_daje_jeden_fragment(self):
        chunks = MarkdownChunker(max_chars=1000, min_chars=5).split(
            "Zwykły tekst bez żadnych nagłówków markdown w środku.", "doc1"
        )
        assert len(chunks) == 1

    def test_dluga_sekcja_dzielona_zdaniowo(self):
        dluga = " ".join(f"To jest zdanie numer {i}." for i in range(100))
        md = f"# Sekcja\n\n{dluga}"
        chunks = MarkdownChunker(max_chars=200, min_chars=10).split(md, "doc1")
        assert len(chunks) > 1
        assert all(c.metadata.get("section") == "Sekcja" for c in chunks)

    def test_pusty_tekst(self):
        assert MarkdownChunker().split("", "doc1") == []


class TestSplitMany:
    def test_dzieli_wiele_dokumentow(self):
        docs = {"a": "Pierwszy dokument.", "b": "Drugi dokument."}
        chunks = SentenceChunker(max_chars=1000).split_many(docs)

        assert len({c.doc_id for c in chunks}) == 2

    def test_indeksy_liczone_per_dokument(self):
        docs = {"a": "x" * 200, "b": "y" * 200}
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).split_many(docs)

        indeksy_a = [c.index for c in chunks if c.doc_id == "a"]
        assert indeksy_a == list(range(len(indeksy_a)))
