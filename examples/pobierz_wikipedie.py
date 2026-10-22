"""Pobiera artykuły z Wikipedii PL przez publiczne API i buduje z nich indeks.

Uruchomienie:
    pip install httpx
    python examples/pobierz_wikipedie.py

Treść Wikipedii jest dostępna na licencji CC BY-SA.
"""

from pathlib import Path

import httpx

from rag_demo import HashingEmbedder, RAGPipeline, SentenceChunker

API = "https://pl.wikipedia.org/w/api.php"

ARTYKULY = [
    "Uczenie maszynowe",
    "Sztuczna sieć neuronowa",
    "Przetwarzanie języka naturalnego",
    "Duży model językowy",
    "Wyszukiwanie informacji",
    "Algorytm",
]

KATALOG = Path(__file__).parent.parent / "data" / "wikipedia"


def pobierz_artykul(client: httpx.Client, tytul: str) -> str | None:
    """Pobiera czysty tekst artykułu. None, gdy artykuł nie istnieje."""
    response = client.get(
        API,
        params={
            "action": "query",
            "format": "json",
            "titles": tytul,
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
        },
        timeout=30.0,
    )
    response.raise_for_status()

    strony = response.json().get("query", {}).get("pages", {})
    for _, strona in strony.items():
        if "missing" in strona:
            return None
        return strona.get("extract", "")
    return None


def main() -> None:
    KATALOG.mkdir(parents=True, exist_ok=True)
    dokumenty: dict[str, str] = {}

    print(f"Pobieram {len(ARTYKULY)} artykułów z Wikipedii PL...\n")

    with httpx.Client(
        headers={"User-Agent": "rag-demo-pl/0.1 (przykład edukacyjny)"}
    ) as client:
        for tytul in ARTYKULY:
            try:
                tresc = pobierz_artykul(client, tytul)
            except httpx.HTTPError as exc:
                print(f"  BŁĄD  {tytul}: {exc}")
                continue

            if not tresc:
                print(f"  BRAK  {tytul}")
                continue

            doc_id = tytul.lower().replace(" ", "_")
            dokumenty[doc_id] = tresc

            sciezka = KATALOG / f"{doc_id}.txt"
            sciezka.write_text(tresc, encoding="utf-8")
            print(f"  OK    {tytul} ({len(tresc):,} znaków)")

    if not dokumenty:
        print("\nNie pobrano żadnego artykułu.")
        return

    print(f"\nZapisano w: {KATALOG}")

    print("\nBuduję indeks...")
    pipeline = RAGPipeline(
        embedder=HashingEmbedder(dimension=512),
        chunker=SentenceChunker(max_chars=700, overlap_sentences=1),
    )
    liczba = pipeline.index(dokumenty, zrodlo="wikipedia-pl", licencja="CC BY-SA")

    print(f"Fragmentów: {liczba}")
    for klucz, wartosc in pipeline.stats().items():
        print(f"  {klucz}: {wartosc}")

    sciezka_indeksu = KATALOG.parent / "wikipedia_index.pkl"
    pipeline.store.save(sciezka_indeksu)
    print(f"\nIndeks zapisany: {sciezka_indeksu}")

    print("\n--- Przykładowe zapytania ---")
    for pytanie in [
        "czym jest sieć neuronowa",
        "jak działa uczenie nadzorowane",
        "przetwarzanie tekstu w języku naturalnym",
    ]:
        print(f"\n> {pytanie}")
        for wynik in pipeline.retrieve(pytanie, top_k=2):
            podglad = wynik.chunk.text[:100].replace("\n", " ")
            print(f"  ({wynik.score:.3f}) [{wynik.chunk.doc_id}] {podglad}...")


if __name__ == "__main__":
    main()
