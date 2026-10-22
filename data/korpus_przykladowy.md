# Przykładowy korpus testowy

Teksty napisane na potrzeby demonstracji, bez ograniczeń licencyjnych.
Tematyka techniczna, żeby zapytania testowe miały sens.

---

## Modele językowe

Duże modele językowe to sieci neuronowe trenowane na obszernych zbiorach tekstu. Uczą się przewidywać kolejny token na podstawie poprzedzającego kontekstu. Ta pozornie prosta reguła prowadzi do zaskakująco złożonych zachowań: modele potrafią streszczać, tłumaczyć i pisać kod, mimo że nie były trenowane bezpośrednio pod te zadania.

Rozmiar kontekstu określa, ile tekstu model może przetworzyć naraz. Wczesne modele miały limit rzędu dwóch tysięcy tokenów. Współczesne rozwiązania obsługują setki tysięcy, co zmienia sposób projektowania systemów: część problemów, które kiedyś wymagały wyszukiwania, można rozwiązać, wrzucając całe dokumenty do kontekstu.

Halucynacja to sytuacja, w której model generuje treść brzmiącą wiarygodnie, ale niezgodną z faktami. Wynika z natury trenowania: model optymalizuje prawdopodobieństwo tekstu, nie jego prawdziwość. Systemy RAG ograniczają ten problem, dostarczając materiał źródłowy i instruując model, żeby się do niego ograniczył.

## Retrieval Augmented Generation

RAG to architektura łącząca wyszukiwanie informacji z generowaniem tekstu. Zamiast polegać wyłącznie na wiedzy zapisanej w wagach modelu, system najpierw znajduje fragmenty dokumentów pasujące do pytania, a następnie przekazuje je modelowi jako kontekst.

Zaletą tego podejścia jest aktualność. Wiedza modelu zatrzymuje się na dacie zakończenia trenowania, natomiast baza dokumentów może być aktualizowana codziennie. Druga zaleta to możliwość cytowania: skoro odpowiedź powstała na podstawie konkretnych fragmentów, można je pokazać użytkownikowi do weryfikacji.

Jakość systemu RAG zależy przede wszystkim od etapu wyszukiwania. Jeśli właściwy fragment nie trafi do kontekstu, żaden model nie odpowie poprawnie. Dlatego kluczową metryką jest recall, odsetek zapytań, dla których trafny materiał znalazł się wśród zwróconych wyników.

## Chunking dokumentów

Chunking to podział dokumentów na fragmenty przed zaindeksowaniem. Decyzja o rozmiarze fragmentu jest kompromisem. Fragmenty zbyt duże zawierają wiele wątków naraz, przez co ich reprezentacja wektorowa staje się uśrednieniem, które nie pasuje dobrze do żadnego konkretnego zapytania. Fragmenty zbyt małe tracą kontekst potrzebny do zrozumienia treści.

Zakładka między fragmentami zmniejsza ryzyko przecięcia istotnej informacji na granicy. Kosztem jest redundancja w indeksie: ten sam tekst pojawia się w kilku fragmentach, zajmując miejsce i mogąc zdominować wyniki wyszukiwania.

Podział świadomy struktury dokumentu daje zwykle lepsze rezultaty niż cięcie co określoną liczbę znaków. Nagłówki, akapity i granice zdań niosą informację o tym, gdzie kończy się jedna myśl, a zaczyna kolejna.

## Osadzenia wektorowe

Embedding to reprezentacja tekstu w postaci wektora liczb rzeczywistych. Model embeddingowy jest trenowany tak, aby teksty o podobnym znaczeniu otrzymywały wektory położone blisko siebie w przestrzeni wielowymiarowej.

Podobieństwo kosinusowe mierzy kąt między wektorami, ignorując ich długość. Dla wektorów znormalizowanych do długości jednostkowej sprowadza się do iloczynu skalarnego, co pozwala policzyć podobieństwo całego korpusu jednym mnożeniem macierzy.

Modele wielojęzyczne umieszczają teksty z różnych języków we wspólnej przestrzeni. Dzięki temu zapytanie po polsku może odnaleźć dokument po angielsku, jeśli oba mówią o tym samym.

## Wyszukiwanie hybrydowe

Wyszukiwanie gęste, oparte na embeddingach, dobrze radzi sobie z parafrazami i synonimami. Ma jednak słaby punkt: rzadkie terminy. Numer katalogowy, akronim czy nazwisko są dla modelu semantycznego niemal szumem, ponieważ nie występowały wystarczająco często w danych treningowych.

BM25 to klasyczny algorytm wyszukiwania leksykalnego. Ocenia dokumenty na podstawie częstości występowania terminów z zapytania, z korektą na długość dokumentu i rzadkość terminu w korpusie. Dosłowne dopasowania znajduje niezawodnie.

Połączenie obu metod daje wyniki lepsze niż każda z osobna. Wymaga jednak normalizacji: wyniki BM25 nie mają górnego ograniczenia, podczas gdy podobieństwo kosinusowe mieści się w przedziale od minus jeden do jeden. Bez sprowadzenia do wspólnej skali jedna metoda zdominowałaby drugą.

## Ewaluacja systemów wyszukiwania

Recall@k mierzy, jaka część trafnych dokumentów znalazła się wśród k pierwszych wyników. Dla systemów RAG jest to metryka najważniejsza, ponieważ materiał nieobecny w kontekście jest dla modelu niedostępny.

Precision@k mówi, jaka część zwróconych wyników była trafna. W kontekście RAG bywa mniej krytyczna niż recall, ponieważ nadmiarowy fragment zwiększa koszt, ale nie uniemożliwia poprawnej odpowiedzi.

Mean Reciprocal Rank nagradza umieszczenie trafnego wyniku wysoko w rankingu. Odwrotność pozycji pierwszego trafienia uśredniona po zapytaniach daje liczbę wrażliwą na kolejność, a nie tylko na obecność.

Wszystkie te metryki wymagają zbioru zapytań z oznaczonymi trafnymi dokumentami. Przygotowanie takiego zbioru jest pracochłonne, ale bez niego optymalizacja systemu sprowadza się do zgadywania.

## Rejestr incydentów

Incydent o identyfikatorze INC-2024-0417 dotyczył nieuprawnionego dostępu do panelu administracyjnego. Przyczyną była podatność CVE-2023-48795, znana jako Terrapin, obniżająca integralność połączenia SSH. Czas reakcji zespołu wyniósł czterdzieści dwie minuty.

Zgłoszenie INC-2024-0892 opisywało wyciek danych konfiguracyjnych przez błędnie ustawiony nagłówek CORS. Oznaczenie w rejestrze podatności: CVE-2024-21626. Skutkiem była rotacja wszystkich kluczy dostępowych w środowisku produkcyjnym.

Postępowanie INC-2025-0103 zostało uruchomione po wykryciu anomalii w ruchu wychodzącym na porcie 8443. Analiza wykazała błędną konfigurację reguły zapory, oznaczoną wewnętrznie jako FW-RULE-7731, a nie faktyczny incydent bezpieczeństwa.
