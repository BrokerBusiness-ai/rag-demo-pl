# rag-demo-pl

![tests](https://github.com/BrokerBusiness-ai/rag-demo-pl/actions/workflows/tests.yml/badge.svg)

System RAG (Retrieval Augmented Generation) dla języka polskiego, od chunkingu, przez wyszukiwanie hybrydowe, po mierzalną ewaluację jakości.

Powstał jako odpowiedź na problem, który wraca w każdym projekcie RAG: łatwo jest złożyć pipeline, który *wygląda*, jakby działał. Trudniej odpowiedzieć na pytanie, czy zmiana strategii chunkingu poprawiła wyniki, czy je pogorszyła. Ten projekt traktuje ewaluację jako element pierwszej klasy, nie dodatek.

**Działa od razu po `git clone`**, domyślny embedder nie wymaga pobierania modelu ani klucza API.

## Szybki start

```bash
git clone https://github.com/BrokerBusiness-ai/rag-demo-pl.git
cd rag-demo-pl
pip install -e ".[dev]"
pytest                       # ~70 testów, bez sieci
python examples/demo.py      # pełne demo
```

```python
from rag_demo import RAGPipeline

pipeline = RAGPipeline()
pipeline.index({
    "dok1": "Python to język programowania wysokiego poziomu...",
    "dok2": "RAG łączy wyszukiwanie z generowaniem odpowiedzi...",
})

odpowiedz = pipeline.query("Czym jest RAG?")
print(odpowiedz.citations())
# [1] dok2, dopasowanie 0.847
```

## Co jest w środku

### Chunking, trzy strategie

Podział tekstu wpływa na jakość wyników bardziej niż wybór modelu embeddingowego. Zbyt duże fragmenty rozmywają sygnał, zbyt małe gubią kontekst.

| Strategia | Kiedy używać |
|---|---|
| `FixedSizeChunker` | Punkt odniesienia. Szybki, ale tnie w środku zdań. |
| `SentenceChunker` | Domyślny wybór. Respektuje granice zdań i **polskie skróty** (np., itd., dr, prof.). |
| `MarkdownChunker` | Dokumentacja i artykuły. Zachowuje hierarchię nagłówków w metadanych. |

Naiwny podział po kropce psuje polszczyznę: „Używamy narzędzi np. Pythona" zostaje pocięte w złym miejscu. `SentenceChunker` zna listę polskich skrótów i tego nie robi.

### Wyszukiwanie hybrydowe

Same embeddingi gubią rzadkie terminy, numery zamówień, akronimy, nazwy własne. Zapytanie „XK-9931-QQ" jest dla modelu semantycznego szumem, a dla BM25 dokładnym trafieniem.

```python
store.search(query, mode="dense")            # tylko embeddingi
store.search(query, mode="lexical")          # tylko BM25
store.search(query, mode="hybrid", alpha=.5) # oba, ważone
```

Wyniki obu metod są normalizowane min-max przed połączeniem, bez tego BM25 (skala nieograniczona) zdominowałby podobieństwo kosinusowe (zakres [-1, 1]).

### Embeddery, wymienne bez zmiany reszty kodu

```python
HashingEmbedder()                 # domyślny, zero zależności, deterministyczny
SentenceTransformerEmbedder()     # realna semantyka, model lokalny
OpenAIEmbedder()                  # API
CachedEmbedder(inner=...)         # dekorator, nie liczy dwa razy tego samego
```

`HashingEmbedder` nie jest modelem semantycznym, nie skojarzy „auta" z „samochodem". Łapie natomiast podobieństwo powierzchniowe i formy fleksyjne, co dla polszczyzny wystarcza do zademonstrowania architektury i pozwala testom działać bez sieci.

### Ewaluacja

```python
from rag_demo import EvalCase, compare_modes, format_comparison

przypadki = [
    EvalCase("jak zainstalować pakiet", relevant_doc_ids={"instalacja"}),
    EvalCase("wymagania systemowe", relevant_doc_ids={"instalacja", "wymagania"}),
]

print(format_comparison(compare_modes(store, przypadki, k=5)))
```

Rzeczywisty wynik z `examples/demo.py` (7 zapytań, k=1, `HashingEmbedder`):

```
Tryb                 Recall  Precision      MRR      Hit
--------------------------------------------------------
dense                 0.857      0.857    0.857    0.857
hybrid(α=1.0)         0.857      0.857    0.857    0.857
hybrid(α=0.5)         0.714      0.714    0.714    0.714
lexical               0.571      0.571    0.571    0.571
```

**Ten wynik przeczy intuicji, i to jest właśnie powód, dla którego warto mierzyć.**

Spodziewałem się przewagi trybu hybrydowego. Wygrało wyszukiwanie gęste, bo domyślny `HashingEmbedder` operuje na n-gramach znakowych, więc częściowo radzi sobie także z identyfikatorami w rodzaju `INC-2024-0892`, dokładnie tam, gdzie miał przegrywać z BM25. Przewaga hybrydy pojawia się dopiero przy embedderze czysto semantycznym, który takich ciągów nie reprezentuje wcale.

Wniosek praktyczny: optymalnej wagi `alpha` nie da się wybrać z góry. Zależy od embeddera i od charakteru zapytań, więc trzeba ją zmierzyć na własnych danych.

Najważniejszą metryką jest **recall@k**, jeśli trafnego fragmentu nie ma w kontekście, model nie odpowie poprawnie, niezależnie od tego, jak jest dobry.

`result.failures()` zwraca przypadki bez trafienia i od nich zaczyna się poprawianie systemu. Demo pokazuje nie tylko ile zapytań każdy tryb psuje, ale **które**, a to one wskazują kierunek zmian:

```
lexical: recall 0.571, nietrafione 3/7
  ✗ skąd wiadomo, że wyszukiwarka zwraca sensowne wyniki
  ✗ model wymyśla fakty, których nie było w danych

dense: recall 0.857, nietrafione 1/7
  ✗ skąd wiadomo, że wyszukiwarka zwraca sensowne wyniki
```

Zapytanie potoczne, bez żadnego terminu technicznego, psuje każdy tryb. To sygnał, że problem jest w warstwie zapytania, nie w wyszukiwaniu, i że następnym krokiem powinno być przepisywanie zapytań (query rewriting), a nie dalsze strojenie `alpha`.

## Podłączenie generatora

Pipeline nie zakłada konkretnego LLM-a. Generator to zwykła funkcja `str -> str`:

```python
def generator(prompt: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

pipeline = RAGPipeline(generator=generator)
odpowiedz = pipeline.query("Jakie są wymagania systemowe?")

print(odpowiedz.answer)
print(odpowiedz.citations())
```

Domyślny prompt instruuje model, żeby odpowiadał wyłącznie z kontekstu i przyznawał się do braku informacji zamiast zmyślać. To najprostsza obrona przed halucynacjami.

## Własny korpus

```python
from pathlib import Path

dokumenty = {
    p.stem: p.read_text(encoding="utf-8")
    for p in Path("moje_dokumenty").glob("*.md")
}
pipeline.index(dokumenty)
pipeline.store.save("index.pkl")
```

W `data/` jest przykładowy korpus po polsku. Skrypt `examples/pobierz_wikipedie.py` pobiera artykuły z Wikipedii PL przez publiczne API.

## Struktura

```
src/rag_demo/
├── chunking.py     # trzy strategie podziału + obsługa polskich skrótów
├── embeddings.py   # Hashing / SentenceTransformer / OpenAI / Cached
├── store.py        # VectorStore, BM25, wyszukiwanie hybrydowe
├── pipeline.py     # spina wszystko, buduje prompt, formatuje cytowania
└── evaluation.py   # recall@k, precision@k, MRR, porównywanie trybów
```

## Ograniczenia

Uczciwie o tym, czego ten projekt nie robi:

- **Skala**, wyszukiwanie to mnożenie macierzy w numpy. Do ~50 tys. fragmentów działa dobrze, wyżej potrzebny FAISS albo baza wektorowa.
- **Brak rerankingu**, nie ma cross-encodera do przesortowania wyników. To kolejny krok, który zwykle daje wyraźny wzrost precyzji.
- **Metryki wymagają etykiet**, ewaluacja potrzebuje ręcznie oznaczonych przypadków testowych. Nie ma tu automatycznej oceny odpowiedzi przez LLM.
- **`HashingEmbedder` nie jest semantyczny**, do realnych zastosowań podstaw `SentenceTransformerEmbedder`.

## Licencja

MIT
