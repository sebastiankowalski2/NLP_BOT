# Discord TODO + NLP bot

Zadanie 1: zarządzanie zadaniami TODO + operacje NLP. Zadanie 2: klasyfikacja tekstu (BoW / TF-IDF / embedding x Linear SVM / Logistic Regression).

## Uruchomienie

Wymagana wersja Pythona: **3.11-3.13** (dla operacji `stem` z `pystempel`).

1. https://discord.com/developers/applications -> New Application -> Bot -> skopiuj token.
2. OAuth2 -> URL Generator: scopes `bot` + `applications.commands`, zaproś bota na serwer.
3. Instalacja (Windows, PowerShell):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download pl_core_news_sm
   python -m spacy download en_core_web_md
   ```
   `pl_core_news_sm` - zadanie 1 (polski), `en_core_web_md` - zadanie 2 (embeddingi z wektorami, angielski).
4. `Copy-Item .env.example .env`, uzupełnij `DISCORD_TOKEN` i `ALBUM_LAST_DIGIT`.
5. `python bot.py`

## Komendy - zadanie 1

`/tasks`, `/add`, `/update`, `/delete`, `/task`
Operacje `/task`: tokens_nltk, tokens_spacy, tokens_compare, clean, normalize, unigram, bigram, trigram, xgram, bow, tfidf, stem, lemma, stopwords

## Zadanie 2: klasyfikacja tekstu

### Dwa osobne zastosowania (dane nie są łączone)
1. **Zadania TODO** (polskie): ręcznie oznaczane klasami **praca / dom / nauka** (pole `label` w tabeli `tasks`).
2. **20 Newsgroups** (angielski, `sklearn.datasets.fetch_20newsgroups`): zbiór do eksperymentów z klasyfikatorami, kategorie:
   `comp.graphics`, `sci.space`, `rec.sport.baseball`, `talk.politics.misc`
   (`subset="all"`, `remove=("headers","footers","quotes")`, pobierany automatycznie przy pierwszym użyciu).

Model z `/classifier` i `/evaluate` jest trenowany na 20 Newsgroups i rozpoznaje tylko te 4 kategorie (po angielsku). Nie rozpoznaje klas praca/dom/nauka.

### Komendy
| Komenda | Działanie |
|---|---|
| `/prepare_data_for_classification` | bez argumentów: lista zadań **bez etykiety**; `task` + `label`: przypisanie lub **nadpisanie** etykiety (praca/dom/nauka). `task` ma autouzupełnianie z aktualną etykietą |
| `/classifier cleaning text method_repr method_class` | klasyfikacja tekstu zapisanym modelem. `cleaning` jest obowiązkowy (lista wyboru, obecnie `clean` = `clean_text_en`), `method_repr`: `bow` / `tfidf` / `embedding`, `method_class`: `svm` / `logistic_regression` |
| `/evaluate method_repr method_class` | macierz pomyłek, ACC, ERR, Precision/Recall/F1 per klasa (one-vs-rest) oraz macro i weighted |

Przykłady:
```
/prepare_data_for_classification task:<wybierz zadanie> label:nauka
/prepare_data_for_classification                      (lista zadań bez etykiety)
/classifier cleaning:clean text:"Space exploration and NASA missions" method_repr:tfidf method_class:svm
/classifier cleaning:clean text:"The pitcher threw a no-hitter" method_repr:embedding method_class:logistic_regression
/evaluate method_repr:tfidf method_class:svm
```
Odpowiedź `/classifier`: `Predicted class: sci.space | repr: tfidf | method: svm | cleaning: clean`.
Prawdopodobieństwo `(p=...)` jest podawane tylko dla `logistic_regression`; `LinearSVC` go nie zwraca (`decision_function` to nie prawdopodobieństwo), więc dla SVM jest pomijane.

Modele są zapisywane w `models/` i wczytywane z dysku - nie są trenowane przy każdym wywołaniu. Gdy modelu brakuje, `/classifier` i `/evaluate` trenują go raz (w osobnym wątku, bot się nie blokuje), zapisują model i wyniki, a przy kolejnych wywołaniach korzystają z zapisu. `/evaluate` nigdy nie zwraca zmyślonych wyników - zawsze są to metryki policzone na zbiorze testowym.

### Uruchomienie trenowania i ewaluacji (6 konfiguracji)
```powershell
python run_experiments.py            # trenuje brakujące modele, zapisuje wyniki
python run_experiments.py --force    # trenuje wszystko od nowa
```
Konfiguracje: BoW+SVM, TF-IDF+SVM, Embedding+SVM, BoW+LogReg, TF-IDF+LogReg, Embedding+LogReg.
Wszystkie używają **tego samego podziału**: `train_test_split` 80/20, `stratify=y`, `random_state=42`.
Wektoryzatory są dopasowywane wyłącznie na zbiorze treningowym (Pipeline) - brak data leakage.
Pierwsze uruchomienie wymaga internetu (pobranie 20 Newsgroups, ok. 14 MB) i modelu `en_core_web_md`.

### Reprezentacje i parametry
| Reprezentacja | Klasa | Parametry (znaczenie) |
|---|---|---|
| **BoW** | `CountVectorizer` | `lowercase=True` (małe litery), `ngram_range=(1,2)` (unigramy i bigramy), `min_df=2` (pomiń cechy w <2 dokumentach), `max_df=0.9` (pomiń cechy w >90% dokumentów), `max_features=20000` (limit słownika), `stop_words="english"` |
| **TF-IDF** | `TfidfVectorizer` | jak BoW oraz `use_idf=True` (waga IDF), `smooth_idf=True` (`ln((1+n)/(1+df))+1`), `sublinear_tf=True` (TF -> `1+ln(TF)`) |
| **Embedding** | spaCy `en_core_web_md`, `doc.vector` | średnia 300-wymiarowych wektorów tokenów; tekst obcinany do 5000 znaków; po embeddingu `StandardScaler`. Kod sprawdza, że model ma wektory oraz że embeddingi nie są puste/zerowe/NaN |

Czyszczenie angielskiego tekstu (`clean_text_en`): małe litery, usunięcie URL-i, e-maili, liczb, interpunkcji, pojedynczych liter i nadmiarowych spacji. Jest to osobna funkcja od polskiej `clean_text` z zadania 1.

### Klasyfikatory
- **Linear SVM**: `LinearSVC(C=1.0, max_iter=10000, random_state=42)`
- **Logistic Regression**: `LogisticRegression(C=1.0, max_iter=2000, random_state=42)`

Embeddingi mają wartości ujemne, dlatego `MultinomialNB` nie jest używany.

### Metryki
Dla każdej konfiguracji: macierz pomyłek, ACC, ERR = 1 - ACC, Precision/Recall/F1 per klasa (one-vs-rest, z TP/FP/FN/TN) oraz średnie macro i weighted (`zero_division=0`).

### Gdzie są wyniki (do analizy w Jupyter Notebook)
Po `python run_experiments.py` w katalogu `results/`:
| Plik | Zawartość |
|---|---|
| `summary.csv` | 6 konfiguracji: ACC, ERR, P/R/F1 macro i weighted, czas trenowania |
| `per_class_metrics.csv` | TP, FP, FN, TN, P, R, F1, ACC/ERR (OvR), support dla każdej klasy |
| `confusion_matrices.json` | macierze pomyłek wszystkich 6 konfiguracji + kolejność klas |
| `<repr>_<clf>.json` | pełny wynik pojedynczej konfiguracji |
| `predictions_<repr>_<clf>.csv` | `test_index`, `y_true`, `y_pred`, `correct` dla zbioru testowego |
| `own_sentences_predictions.csv` | dodatkowa walidacja: 16 własnych zdań (po 4 na klasę) przewidziane przez 6 modeli |
| `experiment_meta.json` | kategorie, indeksy podziału train/test, parametry, wersje bibliotek |

Modele: `models/*.joblib` (nie commituj - dodaj `models/` do `.gitignore`).

## Etykietowanie zadań TODO
`/add` -> `/prepare_data_for_classification` (wybierz zadanie i etykietę `praca`/`dom`/`nauka`) -> `/tasks` pokazuje etykietę jako `#praca`. Starą bazę `tasks.db` bot migruje automatycznie (dodaje kolumnę `label`, żadne dane nie są usuwane).

## Testy
```powershell
pip install -r requirements-dev.txt
python -m pytest
```
Testy używają małego syntetycznego korpusu (nie pobierają 20 Newsgroups) i wymagają `en_core_web_md` oraz `pl_core_news_sm`.

## Publikacja na GitHub

1. Upewnij się, że używasz `.env` lokalnie (nie commituj sekretów).
2. Sprawdź status: `git status`.
3. Dodaj pliki: `git add .`.
4. Zrób commit: `git commit -m "Zadanie 2: klasyfikacja tekstu"`.
5. Wypchnij: `git push`

> Uwaga: jeśli token Discorda był gdziekolwiek upubliczniony, od razu go zresetuj w Discord Developer Portal i wpisz nowy do `.env`.
