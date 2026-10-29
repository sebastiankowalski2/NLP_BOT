"""Operacje NLP na treści zadań (język polski)."""
import os
import re
import sys
import unicodedata
from functools import lru_cache

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- Indywidualny x-gram wyliczany z ostatniej cyfry numeru albumu -----------
# Ostatnia cyfra 4-9 -> x = cyfra; 0-3 -> x = cyfra + 4
ALBUM_LAST_DIGIT = int(os.getenv("ALBUM_LAST_DIGIT", "0"))
X = ALBUM_LAST_DIGIT if ALBUM_LAST_DIGIT >= 4 else ALBUM_LAST_DIGIT + 4
XGRAM_NAME = f"{X}gram"


# --- Zasoby (ładowane leniwie) ----------------------------------------------
@lru_cache(maxsize=1)
def _ensure_nltk() -> None:
    for pkg in ("punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load("pl_core_news_sm")
    except OSError as e:
        raise RuntimeError(
            "Brak modelu spaCy. Uruchom: python -m spacy download pl_core_news_sm"
        ) from e


@lru_cache(maxsize=1)
def _stemmer():
    # pystempel 2.0.0 używa importlib.resources.Resource, które nie działa w Python 3.14+
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Operacja 'stem' wymaga Pythona 3.11-3.13 (pystempel nie wspiera 3.14+)."
        )
    from pystempel import Stemmer  # polski stemmer

    return Stemmer.default()


# --- Tokenizacja -------------------------------------------------------------
def tokens_nltk(text: str) -> list[str]:
    _ensure_nltk()
    return word_tokenize(text, language="polish")


def tokens_spacy(text: str) -> list[str]:
    return [t.text for t in _nlp()(text) if not t.is_space]


def compare_tokenizers(text: str) -> str:
    a, b = tokens_nltk(text), tokens_spacy(text)
    only_a = [t for t in a if t not in b]
    only_b = [t for t in b if t not in a]
    return (
        f"NLTK  ({len(a)}): {a}\n"
        f"spaCy ({len(b)}): {b}\n"
        f"Tylko w NLTK : {only_a or '—'}\n"
        f"Tylko w spaCy: {only_b or '—'}\n"
        f"Identyczne   : {'tak' if a == b else 'nie'}"
    )


# --- Czyszczenie i normalizacja ---------------------------------------------
def clean_text(text: str) -> str:
    """Małe litery, bez URL-i, znaków specjalnych, interpunkcji, liczb,
    pojedynczych znaków i nadmiarowych spacji (polskie litery zostają)."""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)      # URL-e
    text = re.sub(r"\d+", " ", text)               # liczby
    text = re.sub(r"[^\w\s]|_", " ", text)         # interpunkcja i znaki specjalne
    text = re.sub(r"\b\w\b", " ", text)            # pojedyncze znaki
    return re.sub(r"\s+", " ", text).strip()       # białe znaki


_PL_MAP = str.maketrans({"ł": "l", "Ł": "L"})


def normalize_text(text: str, strip_diacritics: bool = True) -> str:
    """NFKC, małe litery, ujednolicenie cudzysłowów/myślników,
    normalizacja białych znaków i (opcjonalnie) usunięcie znaków diakrytycznych."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[“”„«»]", '"', text)
    text = re.sub(r"[‘’‚]", "'", text)
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    if strip_diacritics:
        text = text.translate(_PL_MAP)
        text = "".join(
            c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
        )
    return text


# --- N-gramy -----------------------------------------------------------------
def make_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    tokens = tokens_nltk(clean_text(text))
    return list(ngrams(tokens, n))


# --- Bag of Words / TF-IDF ---------------------------------------------------
BOW_PARAMS = dict(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=1.0,
                  max_features=500, binary=False)
TFIDF_PARAMS = dict(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=1.0,
                    max_features=500, use_idf=True, smooth_idf=True, sublinear_tf=True)

PARAM_DOCS = {
    "lowercase": "zamiana na małe litery przed tokenizacją",
    "ngram_range": "(1, 2) = unigramy i bigramy jako cechy",
    "min_df": "minimalna liczba dokumentów zawierających cechę (1 = bez filtra)",
    "max_df": "maks. udział dokumentów z cechą (1.0 = bez filtra)",
    "max_features": "limit słownika do N najczęstszych cech",
    "binary": "False = liczby wystąpień, True = tylko 0/1",
    "use_idf": "True = mnożenie TF przez IDF (waga rzadkości słowa w korpusie)",
    "smooth_idf": "True = idf = ln((1+n)/(1+df)) + 1, unika dzielenia przez 0",
    "sublinear_tf": "True = TF zastąpione przez 1 + ln(TF), tłumi powtórzenia",
}


def _describe(params: dict) -> str:
    return "\n".join(f"  {k}={v!r} – {PARAM_DOCS[k]}" for k, v in params.items())


def _vectorize(vec, text: str, corpus: list[str]) -> str:
    """Dopasowanie na korpusie (wszystkie zadania); wynik dla wskazanego zadania."""
    docs = [clean_text(d) for d in corpus]
    target = clean_text(text)
    if target not in docs:
        docs.append(target)
    matrix = vec.fit_transform(docs)
    row = matrix[docs.index(target)]
    names = vec.get_feature_names_out()
    pairs = sorted(zip(row.indices, row.data), key=lambda p: -p[1])
    shown = "\n".join(f"  {names[i]!r}: {round(float(v), 4)}" for i, v in pairs) or "  (pusto)"
    return f"Rozmiar słownika: {len(names)}, liczba dokumentów: {len(docs)}\nCechy niezerowe:\n{shown}"


def bow(text: str, corpus: list[str]) -> str:
    return "CountVectorizer – parametry:\n" + _describe(BOW_PARAMS) + "\n\n" + _vectorize(
        CountVectorizer(**BOW_PARAMS), text, corpus)


def tfidf(text: str, corpus: list[str]) -> str:
    return "TfidfVectorizer – parametry:\n" + _describe(TFIDF_PARAMS) + "\n\n" + _vectorize(
        TfidfVectorizer(**TFIDF_PARAMS), text, corpus)


# --- Stemming, lematyzacja, stopwords ---------------------------------------
def stem(text: str) -> list[tuple[str, str]]:
    s = _stemmer()
    return [(t, s(t) or t) for t in tokens_nltk(clean_text(text))]


def lemmatize(text: str) -> list[tuple[str, str]]:
    return [(t.text, t.lemma_) for t in _nlp()(text) if not (t.is_punct or t.is_space)]


def remove_stopwords(text: str) -> list[str]:
    stop = _nlp().Defaults.stop_words  # lista stopwords dla polskiego (spaCy)
    return [t for t in tokens_nltk(text) if t.lower() not in stop and re.search(r"\w", t)]


# --- Dispatcher dla komendy /task -------------------------------------------
def run_operation(op: str, text: str, corpus: list[str]) -> str:
    if op == XGRAM_NAME:
        return f"{X}-gram:\n{make_ngrams(text, X)}"

    match op:
        case "tokens_nltk":
            return str(tokens_nltk(text))
        case "tokens_spacy":
            return str(tokens_spacy(text))
        case "tokens_compare":
            return compare_tokenizers(text)
        case "clean":
            return clean_text(text)
        case "normalize":
            return normalize_text(text)
        case "unigram":
            return str(make_ngrams(text, 1))
        case "bigram":
            return str(make_ngrams(text, 2))
        case "trigram":
            return str(make_ngrams(text, 3))
        case "bow":
            return bow(text, corpus)
        case "tfidf":
            return tfidf(text, corpus)
        case "stem":
            return str(stem(text))
        case "lemma":
            return str(lemmatize(text))
        case "stopwords":
            return f"Oryginał:\n{text}\n\nPo usunięciu stopwords:\n{' '.join(remove_stopwords(text))}"
    raise ValueError(f"Nieznana operacja: {op}")


OPERATIONS = [
    "tokens_nltk", "tokens_spacy", "tokens_compare", "clean", "normalize",
    "unigram", "bigram", "trigram", XGRAM_NAME, "bow", "tfidf",
    "stem", "lemma", "stopwords",
]
