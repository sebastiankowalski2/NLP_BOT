"""Klasyfikacja tekstu (zadanie 2): reprezentacje, trenowanie, predykcja, ewaluacja, zapis wyników.

3 reprezentacje (bow, tfidf, embedding) x 2 klasyfikatory (svm, logistic_regression) = 6 konfiguracji,
wszystkie trenowane na tym samym podziale 80/20 zbioru 20 Newsgroups.
"""
import csv
import hashlib
import json
import logging
import platform
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import classification_data as cdata

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

REPRESENTATIONS = ("bow", "tfidf", "embedding")
CLASSIFIERS = ("svm", "logistic_regression")
CONFIGS = [(r, c) for c in CLASSIFIERS for r in REPRESENTATIONS]  # kolejność 1-6 z zadania

# --- Parametry reprezentacji (opis znaczenia: PARAM_DOCS, powtórzony w README) ------------
BOW_PARAMS = dict(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
                  max_features=20000, stop_words="english")
TFIDF_PARAMS = dict(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
                    max_features=20000, stop_words="english",
                    use_idf=True, smooth_idf=True, sublinear_tf=True)
EMBEDDING_PARAMS = dict(model="en_core_web_md", max_chars=5000)

PARAM_DOCS = {
    "lowercase": "zamiana na małe litery",
    "ngram_range": "(1, 2) = unigramy i bigramy jako cechy",
    "min_df": "pomiń cechy występujące w mniej niż 2 dokumentach (szum, literówki)",
    "max_df": "pomiń cechy obecne w >90% dokumentów (słowa zbyt pospolite)",
    "max_features": "słownik ograniczony do 20000 najczęstszych cech",
    "stop_words": "usuwanie angielskich stopwords wbudowanych w scikit-learn",
    "use_idf": "TF-IDF: waga IDF (rzadkie słowa ważniejsze)",
    "smooth_idf": "idf = ln((1+n)/(1+df)) + 1, bez dzielenia przez 0",
    "sublinear_tf": "TF -> 1 + ln(TF), tłumi powtórzenia słowa w dokumencie",
    "model": "model spaCy z wektorami słów (300 wymiarów); doc.vector = średnia wektorów tokenów",
    "max_chars": "dokument obcinany do 5000 znaków (ogranicza czas liczenia embeddingów)",
}

# Klasyfikatory. Embeddingi mogą mieć wartości ujemne, więc MultinomialNB się nie nadaje -
# używamy modeli liniowych: LinearSVC i LogisticRegression.
SVM_PARAMS = dict(C=1.0, max_iter=10000, random_state=cdata.RANDOM_STATE)
LR_PARAMS = dict(C=1.0, max_iter=2000, random_state=cdata.RANDOM_STATE)

_LOCK = threading.RLock()  # jedno trenowanie naraz (bot wywołuje to z wątków)


# --- Embedding spaCy ------------------------------------------------------------------
_NLP_CACHE: dict = {}
_EMB_CACHE: dict = {}  # (model, hash tekstów) -> macierz; unika liczenia dwa razy dla SVM i LR


def _load_spacy(model: str):
    if model not in _NLP_CACHE:
        import spacy

        try:
            nlp = spacy.load(model)
        except OSError as e:
            raise RuntimeError(
                f"Brak modelu spaCy `{model}`. Zainstaluj: python -m spacy download {model}"
            ) from e
        if nlp.vocab.vectors.shape[0] == 0:
            raise RuntimeError(f"Model `{model}` nie ma wektorów słów (użyj en_core_web_md lub większego).")
        _NLP_CACHE[model] = nlp
    return _NLP_CACHE[model]


def check_embeddings(X: np.ndarray) -> None:
    """Sprawdza kształt, skończoność i niezerowość embeddingów."""
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Nieprawidłowy kształt embeddingów: {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Embeddingi zawierają NaN/inf.")
    if not X.any():
        raise ValueError("Wszystkie embeddingi są zerowe (model bez wektorów lub puste teksty).")


class SpacyEmbedder(BaseEstimator, TransformerMixin):
    """Transformer sklearn: tekst -> doc.vector (spaCy)."""

    def __init__(self, model: str = "en_core_web_md", max_chars: int = 5000):
        self.model = model
        self.max_chars = max_chars

    def fit(self, X, y=None):
        _load_spacy(self.model)  # wczesne sprawdzenie, czy model istnieje i ma wektory
        return self

    def transform(self, X):
        texts = [t[: self.max_chars] for t in X]
        key = (self.model, hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest())
        if key not in _EMB_CACHE:
            nlp = _load_spacy(self.model)
            vecs = [d.vector for d in nlp.pipe(texts, batch_size=64, disable=nlp.pipe_names)]
            _EMB_CACHE[key] = np.vstack(vecs).astype(np.float32)
        out = _EMB_CACHE[key]
        check_embeddings(out)
        return out


# --- Pipeline ------------------------------------------------------------------------
def validate_config(method_repr: str, method_class: str) -> None:
    if method_repr not in REPRESENTATIONS:
        raise ValueError(f"Nieznana reprezentacja `{method_repr}`. Dozwolone: {', '.join(REPRESENTATIONS)}.")
    if method_class not in CLASSIFIERS:
        raise ValueError(f"Nieznany klasyfikator `{method_class}`. Dozwolone: {', '.join(CLASSIFIERS)}.")


def build_pipeline(method_repr: str, method_class: str) -> Pipeline:
    """Pipeline = reprezentacja + klasyfikator. Wektoryzator uczy się TYLKO w fit() na danych
    treningowych, a dane testowe są tylko transformowane (brak data leakage)."""
    validate_config(method_repr, method_class)
    steps = []
    if method_repr == "bow":
        steps.append(("repr", CountVectorizer(**BOW_PARAMS)))
    elif method_repr == "tfidf":
        steps.append(("repr", TfidfVectorizer(**TFIDF_PARAMS)))
    else:
        steps.append(("repr", SpacyEmbedder(**EMBEDDING_PARAMS)))
        steps.append(("scale", StandardScaler()))  # embeddingi: skala i wartości ujemne
    clf = LinearSVC(**SVM_PARAMS) if method_class == "svm" else LogisticRegression(**LR_PARAMS)
    steps.append(("clf", clf))
    return Pipeline(steps)


# --- Metryki i macierz pomyłek ---------------------------------------------------------
def compute_metrics(y_true, y_pred, labels) -> dict:
    """Macierz pomyłek + ACC, ERR oraz P/R/F1 (one-vs-rest dla każdej klasy, średnie macro i weighted)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    avg = {}
    for kind in ("macro", "weighted"):
        pk, rk, fk, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=kind, zero_division=0
        )
        avg[kind] = {"precision": float(pk), "recall": float(rk), "f1": float(fk)}

    per_class = {}
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = total - tp - fp - fn
        per_class[name] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "acc_ovr": (tp + tn) / total, "err_ovr": (fp + fn) / total,
            "support": int(s[i]),
        }
    return {
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "error_rate": 1.0 - acc,  # ERR = 1 - ACC
        "macro": avg["macro"],
        "weighted": avg["weighted"],
        "per_class": per_class,
        "n_test": total,
    }


# --- Ścieżki, zapis i odczyt -------------------------------------------------------------
def _paths(method_repr, method_class, models_dir, results_dir):
    name = f"{method_repr}_{method_class}"
    return Path(models_dir) / f"{name}.joblib", Path(results_dir) / f"{name}.json", name


_SPLIT_CACHE: dict = {}


def get_split() -> cdata.Split:
    if "split" not in _SPLIT_CACHE:
        _SPLIT_CACHE["split"] = cdata.prepare_split()
    return _SPLIT_CACHE["split"]


def train_and_evaluate(method_repr, method_class, split=None, models_dir=None, results_dir=None,
                       force=False) -> dict:
    """Trenuje (jeśli brak modelu lub force=True), ewaluuje na zbiorze testowym i zapisuje wyniki.

    Gdy model i wyniki już istnieją, nic nie trenuje - wczytuje zapisany wynik.
    """
    validate_config(method_repr, method_class)
    models_dir, results_dir = Path(models_dir or MODELS_DIR), Path(results_dir or RESULTS_DIR)
    model_path, result_path, name = _paths(method_repr, method_class, models_dir, results_dir)
    with _LOCK:
        if not force and model_path.exists() and result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        split = split or get_split()
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        pipe = build_pipeline(method_repr, method_class)
        t0 = time.time()
        pipe.fit(split.X_train, split.y_train)
        train_seconds = time.time() - t0
        y_pred = pipe.predict(split.X_test)

        result = compute_metrics(split.y_test, y_pred, split.labels)
        result.update(representation=method_repr, classifier=method_class,
                      train_seconds=round(train_seconds, 2), n_train=len(split.X_train))
        joblib.dump({"pipeline": pipe, "labels": split.labels, "sklearn": sklearn.__version__},
                    model_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_predictions(results_dir / f"predictions_{name}.csv", split, y_pred)
        _MODEL_CACHE.pop(str(model_path), None)
        return result


def _save_predictions(path: Path, split, y_pred) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "y_true", "y_pred", "correct"])
        for i, t, p in zip(split.idx_test, split.y_test, y_pred):
            w.writerow([i, t, p, int(t == p)])


_MODEL_CACHE: dict = {}


def load_model(method_repr, method_class, models_dir=None) -> dict:
    """Wczytuje zapisany model (z cache w pamięci procesu)."""
    path, _, _ = _paths(method_repr, method_class, models_dir or MODELS_DIR, RESULTS_DIR)
    key = str(path)
    if key not in _MODEL_CACHE:
        if not path.exists():
            raise FileNotFoundError(f"Brak modelu {method_repr}/{method_class}.")
        _MODEL_CACHE[key] = joblib.load(path)
    return _MODEL_CACHE[key]


# --- Predykcja --------------------------------------------------------------------------
def classify(text: str, method_repr: str, method_class: str, cleaning: str = "clean",
             models_dir=None, results_dir=None, train_if_missing: bool = True) -> dict:
    """Czyści tekst, stosuje zapisany model i zwraca klasę (+ prawdopodobieństwo tylko dla LR)."""
    validate_config(method_repr, method_class)
    if cleaning not in cdata.CLEANING_METHODS:
        raise ValueError(f"Nieznane czyszczenie `{cleaning}`. Dozwolone: {', '.join(cdata.CLEANING_METHODS)}.")
    cleaned = cdata.CLEANING_METHODS[cleaning](text)
    if not cleaned:
        raise ValueError("Po czyszczeniu tekst jest pusty (model zna tylko angielski tekst z liter).")

    models_dir = Path(models_dir or MODELS_DIR)
    model_path = _paths(method_repr, method_class, models_dir, results_dir or RESULTS_DIR)[0]
    trained_now = False
    if not model_path.exists():
        if not train_if_missing:
            raise FileNotFoundError(f"Brak modelu {method_repr}/{method_class}. Uruchom: python run_experiments.py")
        train_and_evaluate(method_repr, method_class, models_dir=models_dir, results_dir=results_dir)
        trained_now = True

    pipe = load_model(method_repr, method_class, models_dir)["pipeline"]
    pred = str(pipe.predict([cleaned])[0])
    prob = None
    # LinearSVC nie zwraca prawdopodobieństw (decision_function to nie prawdopodobieństwo) - pomijamy.
    if method_class == "logistic_regression":
        probs = pipe.predict_proba([cleaned])[0]
        prob = float(probs[list(pipe.classes_).index(pred)])
    return {"predicted": pred, "probability": prob, "representation": method_repr,
            "classifier": method_class, "cleaning": cleaning, "cleaned_text": cleaned,
            "trained_now": trained_now}


def format_classification(r: dict) -> str:
    p = f" (p={r['probability']:.2f})" if r["probability"] is not None else ""
    return (f"Predicted class: **{r['predicted']}**{p} | repr: {r['representation']} | "
            f"method: {r['classifier']} | cleaning: {r['cleaning']}")


# --- Formatowanie wyniku ewaluacji (Discord) --------------------------------------------------
def format_evaluation(r: dict) -> str:
    labels = r["labels"]
    letters = [chr(ord("A") + i) for i in range(len(labels))]
    lines = [f"Konfiguracja: {r['representation']} + {r['classifier']}  (zbiór testowy: {r['n_test']})",
             "Klasy: " + ", ".join(f"{a}={n}" for a, n in zip(letters, labels)),
             "", "Macierz pomyłek (wiersze = prawdziwa, kolumny = przewidziana):",
             "      " + " ".join(f"{a:>5}" for a in letters)]
    for a, row in zip(letters, r["confusion_matrix"]):
        lines.append(f"{a:>5} " + " ".join(f"{v:>5}" for v in row))
    lines += ["", f"ACC = {r['accuracy']:.4f}   ERR = {r['error_rate']:.4f}", "",
              f"{'klasa':<20}{'P':>8}{'R':>8}{'F1':>8}{'supp':>7}"]
    for n, m in r["per_class"].items():
        lines.append(f"{n:<20}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}{m['support']:>7}")
    for kind in ("macro", "weighted"):
        m = r[kind]
        lines.append(f"{kind + ' avg':<20}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}")
    return "\n".join(lines)


def load_result(method_repr, method_class, results_dir=None):
    validate_config(method_repr, method_class)
    path = _paths(method_repr, method_class, MODELS_DIR, results_dir or RESULTS_DIR)[1]
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- Własne zdania (dodatkowa walidacja spoza zbioru) -----------------------------------------
OWN_SENTENCES = [
    ("How do I render a 3D scene with realistic shadows using OpenGL?", "comp.graphics"),
    ("Which image format keeps more detail, JPEG or PNG, when I resize a picture?", "comp.graphics"),
    ("My graphics card draws polygons very slowly in the new rendering software.", "comp.graphics"),
    ("I need a program that converts bitmap images to vector graphics.", "comp.graphics"),
    ("NASA plans to launch a new probe to study the moons of Jupiter.", "sci.space"),
    ("The shuttle crew repaired the orbiting telescope during a spacewalk.", "sci.space"),
    ("Astronomers detected a distant planet orbiting a nearby star.", "sci.space"),
    ("How much fuel does a rocket need to reach low earth orbit?", "sci.space"),
    ("The pitcher struck out ten batters and the team won in the ninth inning.", "rec.sport.baseball"),
    ("Our shortstop hit a home run in the playoff game last night.", "rec.sport.baseball"),
    ("The manager changed the batting order before the season opener.", "rec.sport.baseball"),
    ("Who has the best earned run average in the league this year?", "rec.sport.baseball"),
    ("The senator proposed a new law about health care and taxes.", "talk.politics.misc"),
    ("Voters will choose the next president at the coming election.", "talk.politics.misc"),
    ("The government debated the budget and the role of the Supreme Court.", "talk.politics.misc"),
    ("Congress is arguing about immigration policy and campaign funding.", "talk.politics.misc"),
]


def validate_own_sentences(models_dir=None, results_dir=None) -> list:
    """Przewiduje klasy własnych zdań wszystkimi 6 modelami i zapisuje CSV (do omówienia w notebooku)."""
    models_dir, results_dir = Path(models_dir or MODELS_DIR), Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_repr, method_class in CONFIGS:
        for text, expected in OWN_SENTENCES:
            r = classify(text, method_repr, method_class, models_dir=models_dir,
                         results_dir=results_dir, train_if_missing=False)
            rows.append({"sentence": text, "expected": expected, "representation": method_repr,
                         "classifier": method_class, "predicted": r["predicted"],
                         "correct": int(r["predicted"] == expected),
                         "probability": "" if r["probability"] is None else round(r["probability"], 4)})
    _write_csv(results_dir / "own_sentences_predictions.csv", rows)
    return rows


# --- Zbiorcze wyniki 6 konfiguracji ------------------------------------------------------------
def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary(results_dir=None) -> list:
    """Buduje summary.csv, per_class_metrics.csv i confusion_matrices.json z zapisanych wyników."""
    results_dir = Path(results_dir or RESULTS_DIR)
    summary, per_class, matrices = [], [], {}
    for method_repr, method_class in CONFIGS:
        r = load_result(method_repr, method_class, results_dir)
        if r is None:
            continue
        summary.append({
            "representation": method_repr, "classifier": method_class,
            "accuracy": round(r["accuracy"], 4), "error_rate": round(r["error_rate"], 4),
            "precision_macro": round(r["macro"]["precision"], 4),
            "recall_macro": round(r["macro"]["recall"], 4), "f1_macro": round(r["macro"]["f1"], 4),
            "precision_weighted": round(r["weighted"]["precision"], 4),
            "recall_weighted": round(r["weighted"]["recall"], 4),
            "f1_weighted": round(r["weighted"]["f1"], 4),
            "n_train": r.get("n_train", ""), "n_test": r["n_test"],
            "train_seconds": r.get("train_seconds", ""),
        })
        for cls, m in r["per_class"].items():
            per_class.append({"representation": method_repr, "classifier": method_class,
                              "class": cls, **{k: (round(v, 4) if isinstance(v, float) else v)
                                               for k, v in m.items()}})
        matrices[f"{method_repr}_{method_class}"] = {"labels": r["labels"],
                                                     "matrix": r["confusion_matrix"]}
    _write_csv(results_dir / "summary.csv", summary)
    _write_csv(results_dir / "per_class_metrics.csv", per_class)
    (results_dir / "confusion_matrices.json").write_text(
        json.dumps(matrices, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def format_summary(rows: list) -> str:
    head = f"{'repr':<11}{'classifier':<20}{'ACC':>8}{'ERR':>8}{'F1 macro':>10}{'F1 wtd':>9}"
    body = [f"{r['representation']:<11}{r['classifier']:<20}{r['accuracy']:>8.4f}{r['error_rate']:>8.4f}"
            f"{r['f1_macro']:>10.4f}{r['f1_weighted']:>9.4f}" for r in rows]
    return "\n".join([head] + body)


def run_all(force=False, models_dir=None, results_dir=None, split=None, own_sentences=True) -> list:
    """Trenuje i ewaluuje wszystkie 6 konfiguracji na jednym podziale i zapisuje komplet wyników."""
    models_dir, results_dir = Path(models_dir or MODELS_DIR), Path(results_dir or RESULTS_DIR)
    split = split or get_split()
    for method_repr, method_class in CONFIGS:
        log.info("Konfiguracja: %s + %s", method_repr, method_class)
        train_and_evaluate(method_repr, method_class, split=split, models_dir=models_dir,
                           results_dir=results_dir, force=force)
    summary = write_summary(results_dir)
    if own_sentences:
        validate_own_sentences(models_dir, results_dir)
    write_meta(split, results_dir)
    return summary


def write_meta(split, results_dir=None) -> None:
    """experiment_meta.json: dane potrzebne do odtworzenia podziału i konfiguracji."""
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "20 Newsgroups (sklearn.datasets.fetch_20newsgroups, subset=all, "
                   "remove=headers/footers/quotes)",
        "categories": cdata.CATEGORIES,
        "split": {"test_size": cdata.TEST_SIZE, "stratify": True, "random_state": cdata.RANDOM_STATE,
                  "n_train": len(split.X_train), "n_test": len(split.X_test),
                  "n_dropped_empty_documents": split.n_dropped_empty,
                  "idx_train": split.idx_train, "idx_test": split.idx_test},
        "params": {"bow": BOW_PARAMS, "tfidf": TFIDF_PARAMS, "embedding": EMBEDDING_PARAMS,
                   "svm": SVM_PARAMS, "logistic_regression": LR_PARAMS,
                   "embedding_extra": "StandardScaler po embeddingach"},
        "param_docs": PARAM_DOCS,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__},
    }
    try:
        import spacy
        meta["versions"]["spacy"] = spacy.__version__
        nlp = _NLP_CACHE.get(EMBEDDING_PARAMS["model"])
        if nlp is not None:
            meta["versions"]["spacy_model"] = f"{nlp.meta['name']} {nlp.meta['version']}"
    except Exception:  # metadane pomocnicze - brak nie może psuć zapisu wyników
        pass
    (results_dir / "experiment_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
