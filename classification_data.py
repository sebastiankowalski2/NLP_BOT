"""Dane do klasyfikacji (zadanie 2): 20 Newsgroups, czyszczenie angielskiego tekstu, podział 80/20.

Uwaga: zbiór 20 Newsgroups (angielski) jest używany WYŁĄCZNIE do eksperymentów z klasyfikatorami.
Zadania TODO (polskie, etykiety praca/dom/nauka) to osobny zbiór i nie są z nim łączone.
"""
import re
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

CATEGORIES = ["comp.graphics", "sci.space", "rec.sport.baseball", "talk.politics.misc"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


# --- Czyszczenie (dedykowane dla angielskiego; polska clean_text z zadania 1 usuwa
# --- znaki spoza polskiego alfabetu w inny sposób i nie nadaje się do tego zbioru) --
def clean_text_en(text: str) -> str:
    """Małe litery, bez URL-i, e-maili, liczb, interpunkcji, pojedynczych liter i nadmiarowych spacji."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URL-e
    text = re.sub(r"\S+@\S+", " ", text)                # adresy e-mail
    text = re.sub(r"\d+", " ", text)                    # liczby
    text = re.sub(r"[^a-z\s]", " ", text)               # interpunkcja i znaki specjalne
    text = re.sub(r"\b[a-z]\b", " ", text)              # pojedyncze litery
    return re.sub(r"\s+", " ", text).strip()            # białe znaki


# Dostępne warianty czyszczenia (wymagany parametr `cleaning` komendy /classifier)
CLEANING_METHODS = {"clean": clean_text_en}


@dataclass
class Split:
    X_train: list
    X_test: list
    y_train: list
    y_test: list
    idx_train: list  # indeksy w zbiorze po czyszczeniu (do odtworzenia podziału)
    idx_test: list
    labels: list     # kolejność klas w macierzach pomyłek
    n_dropped_empty: int = 0


def load_20newsgroups():
    """Pobiera zbiór przez fetch_20newsgroups (cache w ~/scikit_learn_data).

    Zwraca (teksty_po_czyszczeniu, etykiety_tekstowe, liczba_odrzuconych_pustych).
    Dokumenty puste po usunięciu nagłówków/stopek/cytatów i po czyszczeniu są odrzucane.
    """
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    names = [data.target_names[i] for i in data.target]
    return _clean_and_filter(data.data, names)


def _clean_and_filter(raw_texts, names):
    texts, labels, dropped = [], [], 0
    for t, n in zip(raw_texts, names):
        c = clean_text_en(t)
        if c:
            texts.append(c)
            labels.append(n)
        else:
            dropped += 1
    return texts, labels, dropped


def make_split(texts, labels, dropped: int = 0) -> Split:
    """80/20, stratify=y, random_state=42 — jeden podział dla wszystkich sześciu konfiguracji."""
    idx = list(range(len(texts)))
    idx_tr, idx_te = train_test_split(
        idx, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    return Split(
        X_train=[texts[i] for i in idx_tr],
        X_test=[texts[i] for i in idx_te],
        y_train=[labels[i] for i in idx_tr],
        y_test=[labels[i] for i in idx_te],
        idx_train=idx_tr,
        idx_test=idx_te,
        labels=sorted(set(labels)),
        n_dropped_empty=dropped,
    )


def prepare_split() -> Split:
    texts, labels, dropped = load_20newsgroups()
    return make_split(texts, labels, dropped)
