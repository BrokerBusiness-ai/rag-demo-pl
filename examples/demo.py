"""Demo end-to-end: indeksowanie, wyszukiwanie, porównanie trybów, ewaluacja.

Uruchomienie:
    python examples/demo.py

Nie wymaga kluczy API ani pobierania modeli.
"""

from pathlib import Path

from rag_demo import (
    EvalCase,
    HashingEmbedder,
    MarkdownChunker,
    RAGPipeline,
    compare_modes,
    evaluate,
    format_comparison,
)

KORPUS = Path(__file__).parent.parent / "data" / "korpus_przykladowy.md"


def wczytaj_korpus() -> dict[str, str]:
    """Wczytuje korpus i dzieli go na dokumenty po nagłówkach drugiego poziomu."""
    tekst = KORPUS.read_text(encoding="utf-8")

    dokumenty: dict[str, str] = {}
    biezacy_tytul = None
    biezaca_tresc: list[str] = []

    for linia in tekst.splitlines():
        if linia.startswith("## "):
            if biezacy_tytul:
                dokumenty[biezacy_tytul] = "\n".join(biezaca_tresc).strip()
            biezacy_tytul = linia[3:].strip().lower().replace(" ", "_")
            biezaca_tresc = []
        elif biezacy_tytul:
            biezaca_tresc.append(linia)

    if biezacy_tytul:
        dokumenty[biezacy_tytul] = "\n".join(biezaca_tresc).strip()

    return dokumenty


def sekcja(tytul: str) -> None:
    print(f"\n{'=' * 62}\n{tytul}\n{'=' * 62}")


def main() -> None:
    dokumenty = wczytaj_korpus()

    sekcja("1. Indeksowanie")
    pipeline = RAGPipeline(
        embedder=HashingEmbedder(dimension=512),
        chunker=MarkdownChunker(max_chars=600, min_chars=100),
        top_k=3,
    )
    liczba = pipeline.index(dokumenty, zrodlo="korpus_przykladowy")

    print(f"Wczytano dokumentów: {len(dokumenty)}")
    print(f"Utworzono fragmentów: {liczba}\n")
    for klucz, wartosc in pipeline.stats().items():
        print(f"  {klucz}: {wartosc}")

    sekcja("2. Wyszukiwanie, trzy tryby, to samo pytanie")
    pytanie = "dlaczego BM25 znajduje rzadkie terminy lepiej niż embeddingi"

    for tryb in ("dense", "lexical", "hybrid"):
        print(f"\n--- tryb: {tryb} ---")
        wyniki = pipeline.store.search(pytanie, top_k=2, mode=tryb)
        for wynik in wyniki:
            print(f"  {wynik}")

    sekcja("3. Pełne zapytanie z cytowaniami")
    odpowiedz = pipeline.query("Czym różni się recall od precision?")

    print(f"Pytanie: {odpowiedz.question}\n")
    print("Znalezione fragmenty:")
    for wynik in odpowiedz.sources:
        print(f"  {wynik}")

    print(f"\n{odpowiedz.citations()}")
    print(f"\nMetadane: {odpowiedz.metadata}")

    sekcja("4. Prompt wysyłany do modelu")
    podglad = odpowiedz.prompt[:600]
    print(podglad + ("..." if len(odpowiedz.prompt) > 600 else ""))

    sekcja("5. Ewaluacja, który tryb działa najlepiej")

    # k=1 celowo. Przy k=3 każdy tryb trafia i porównanie nic nie pokazuje -
    # dopiero wymóg trafienia w pierwszy wynik różnicuje metody.
    przypadki = [
        # Rzadkie identyfikatory: dosłowne dopasowanie, przewaga BM25.
        EvalCase(
            "INC-2024-0892",
            {"rejestr_incydentów"},
            note="Identyfikator incydentu, model semantyczny go nie zna",
        ),
        EvalCase(
            "CVE-2023-48795 Terrapin",
            {"rejestr_incydentów"},
            note="Oznaczenie CVE, rzadki termin",
        ),
        EvalCase(
            "FW-RULE-7731",
            {"rejestr_incydentów"},
            note="Wewnętrzne oznaczenie reguły zapory",
        ),
        # Parafrazy: brak wspólnych słów z dokumentem, przewaga embeddingów.
        EvalCase(
            "w jaki sposób ciąć tekst na kawałki przed indeksowaniem",
            {"chunking_dokumentów"},
            note="Parafraza bez terminu 'chunking', przewaga gęstego",
        ),
        EvalCase(
            "skąd wiadomo, że wyszukiwarka zwraca sensowne wyniki",
            {"ewaluacja_systemów_wyszukiwania"},
            note="Opis potoczny zamiast nazw metryk",
        ),
        # Mieszane: termin techniczny plus kontekst opisowy.
        EvalCase(
            "dlaczego embeddingi gubią numery katalogowe",
            {"wyszukiwanie_hybrydowe"},
            note="Termin + parafraza, tu hybryda powinna wygrać",
        ),
        EvalCase(
            "model wymyśla fakty, których nie było w danych",
            {"modele_językowe"},
            note="Opis halucynacji bez użycia słowa 'halucynacja'",
        ),
    ]

    wyniki = compare_modes(pipeline.store, przypadki, k=1)
    print(format_comparison(wyniki))

    najlepszy = wyniki[0]
    print(f"\nNajlepszy tryb: {najlepszy.mode}")

    # Pokazujemy, że tryby różnią się nie tylko wynikiem łącznym, ale tym,
    # KTÓRE zapytania psują, to jest argument za hybrydą.
    print("\n--- Co psuje każdy tryb (k=1) ---")
    for tryb, alpha in (("lexical", 0.0), ("dense", 1.0), ("hybrid", 0.5)):
        wynik = evaluate(pipeline.store, przypadki, k=1, mode=tryb, alpha=alpha)
        etykieta = tryb if tryb != "hybrid" else f"hybrid(α={alpha})"
        nietrafione = wynik.failures()
        print(f"\n{etykieta}: recall {wynik.recall_at_k:.3f}, "
              f"nietrafione {len(nietrafione)}/{len(przypadki)}")
        for case in nietrafione:
            print(f"  ✗ {case['query'][:52]}")
            print(f"    → {case['note']}")

    sekcja("6. Podłączenie generatora")
    print(
        "Pipeline nie zakłada konkretnego LLM-a. Generator to funkcja str -> str:\n"
    )
    print("""    def generator(prompt: str) -> str:
        return anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        ).content[0].text

    pipeline = RAGPipeline(generator=generator)
    print(pipeline.query("pytanie").answer)""")

    def generator_zastepczy(prompt: str) -> str:
        liczba_fragmentow = prompt.count("\n[")
        return (
            f"[Generator zastępczy] Otrzymałem {len(prompt)} znaków promptu "
            f"zawierającego {liczba_fragmentow + 1} fragmentów. "
            "Podstaw prawdziwy LLM, żeby dostać właściwą odpowiedź."
        )

    odp = pipeline.query("Czym jest RAG?", generator=generator_zastepczy)
    print(f"\n{odp.answer}")


if __name__ == "__main__":
    main()
