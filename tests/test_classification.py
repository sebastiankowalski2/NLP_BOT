import csv
import json
import math

import numpy as np
import pytest

import classification as clf
import classification_data as cdata
from conftest import synthetic_corpus

ALL_CONFIGS = clf.CONFIGS


# --- dane, czyszczenie, podział ---------------------------------------------------------
def test_clean_text_en():
    out = cdata.clean_text_en("Hello, World!! Visit http://x.org or mail a.b@c.com. 42 apples & I x")
    assert out == "hello world visit or mail apples"


def test_load_20newsgroups_filters_empty(monkeypatch):
    """Logika ładowania bez internetu: fetch_20newsgroups podmieniony na atrapę."""
    import sklearn.datasets as skd

    class FakeBunch:
        data = ["Nice  GPU!!", "", "  123 ... ", "Space shuttle"]
        target = [0, 0, 1, 1]
        target_names = ["comp.graphics", "sci.space"]

    called = {}

    def fake_fetch(**kw):
        called.update(kw)
        return FakeBunch()

    monkeypatch.setattr(skd, "fetch_20newsgroups", fake_fetch)
    texts, labels, dropped = cdata.load_20newsgroups()
    assert texts == ["nice gpu", "space shuttle"] and dropped == 2
    assert called["categories"] == cdata.CATEGORIES
    assert called["subset"] == "all" and called["remove"] == ("headers", "footers", "quotes")
    assert len(cdata.CATEGORIES) == 4


def test_split_80_20_stratified_deterministic(split):
    n = len(split.X_train) + len(split.X_test)
    assert len(split.X_test) == pytest.approx(0.2 * n, abs=1)
    for c in split.labels:
        share_all = (split.y_train.count(c) + split.y_test.count(c)) / n
        assert split.y_test.count(c) / len(split.X_test) == pytest.approx(share_all, abs=0.03)
    assert not set(split.idx_train) & set(split.idx_test)
    texts, labels = synthetic_corpus()
    again = cdata.make_split(texts, labels)
    assert again.idx_test == split.idx_test and again.idx_train == split.idx_train


# --- metryki -----------------------------------------------------------------------------
def test_compute_metrics_hand_example():
    y_true = ["a"] * 3 + ["b"] * 3
    y_pred = ["a", "a", "b", "b", "b", "b"]
    m = clf.compute_metrics(y_true, y_pred, ["a", "b"])
    assert m["confusion_matrix"] == [[2, 1], [0, 3]]
    assert m["accuracy"] == pytest.approx(5 / 6) and m["error_rate"] == pytest.approx(1 / 6)
    a = m["per_class"]["a"]
    assert (a["TP"], a["FP"], a["FN"], a["TN"]) == (2, 0, 1, 3)
    assert a["precision"] == 1.0 and a["recall"] == pytest.approx(2 / 3)
    assert a["f1"] == pytest.approx(0.8)
    assert m["macro"]["f1"] == pytest.approx((0.8 + 6 / 7) / 2)


def test_compute_metrics_class_without_predictions():
    m = clf.compute_metrics(["a", "b", "c"], ["a", "a", "a"], ["a", "b", "c"])
    assert m["per_class"]["b"]["precision"] == 0.0 and m["per_class"]["c"]["f1"] == 0.0
    assert all(math.isfinite(v) for v in m["macro"].values())


# --- sześć konfiguracji ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory, split):
    base = tmp_path_factory.mktemp("exp")
    d = {"models_dir": base / "models", "results_dir": base / "results"}
    results = {}
    for r, c in ALL_CONFIGS:
        results[(r, c)] = clf.train_and_evaluate(r, c, split=split, **d)
    return d, results


@pytest.mark.parametrize("method_repr,method_class", ALL_CONFIGS)
def test_each_configuration(trained, split, method_repr, method_class):
    d, results = trained
    r = results[(method_repr, method_class)]
    cm = np.array(r["confusion_matrix"])
    assert cm.shape == (4, 4) and cm.sum() == len(split.X_test)
    assert r["accuracy"] + r["error_rate"] == pytest.approx(1.0)
    assert r["accuracy"] > 0.8  # korpus syntetyczny jest łatwy; to tylko test poprawności potoku
    assert set(r["per_class"]) == set(split.labels)
    assert (d["models_dir"] / f"{method_repr}_{method_class}.joblib").exists()
    rows = list(csv.DictReader(open(d["results_dir"] / f"predictions_{method_repr}_{method_class}.csv")))
    assert len(rows) == len(split.X_test)


def test_cached_result_is_not_retrained(trained, split):
    d, results = trained
    again = clf.train_and_evaluate("bow", "svm", split=None, **d)  # split=None: nie wolno go ładować
    assert again["accuracy"] == results[("bow", "svm")]["accuracy"]


def test_no_vectorizer_leakage(split):
    pipe = clf.build_pipeline("bow", "svm")
    X_tr = split.X_train
    pipe.fit(X_tr, split.y_train)
    vocab = pipe.named_steps["repr"].vocabulary_
    assert "zzzonlyintest" not in vocab
    pipe.predict(["zzzonlyintest zzzonlyintest orbit"])
    assert "zzzonlyintest" not in pipe.named_steps["repr"].vocabulary_  # transform nie zmienia słownika
    train_vocab_terms = {w for t in X_tr for w in t.split()}
    unigrams = {t for t in vocab if " " not in t}
    assert unigrams <= train_vocab_terms


# --- embeddingi ------------------------------------------------------------------------------------
def test_embedding_shapes_nonzero():
    emb = clf.SpacyEmbedder().fit(None)
    X = emb.transform(["space shuttle orbit", "baseball pitcher inning"])
    assert X.shape == (2, 300) and X.any() and np.isfinite(X).all()
    assert (X < 0).any()  # są wartości ujemne -> MultinomialNB się nie nadaje


def test_embedding_checks():
    with pytest.raises(ValueError):
        clf.check_embeddings(np.zeros((3, 300)))
    with pytest.raises(RuntimeError):
        clf.SpacyEmbedder(model="en_core_web_nonexistent").fit(None)


# --- zapis/odczyt i predykcja -------------------------------------------------------------------------
@pytest.mark.parametrize("method_repr,method_class", ALL_CONFIGS)
def test_save_load_predict(trained, split, method_repr, method_class):
    d, _ = trained
    model = clf.load_model(method_repr, method_class, d["models_dir"])
    expected = model["pipeline"].predict(split.X_test[:5])
    import joblib
    fresh = joblib.load(d["models_dir"] / f"{method_repr}_{method_class}.joblib")["pipeline"]
    assert list(fresh.predict(split.X_test[:5])) == list(expected)
    r = clf.classify("The rocket will reach orbit around the moon", method_repr, method_class,
                     models_dir=d["models_dir"], results_dir=d["results_dir"], train_if_missing=False)
    assert r["predicted"] in split.labels
    if method_class == "svm":
        assert r["probability"] is None  # decision_function to nie prawdopodobieństwo
    else:
        assert 0.0 <= r["probability"] <= 1.0


def test_classify_invalid_arguments(trained):
    d, _ = trained
    kw = dict(models_dir=d["models_dir"], results_dir=d["results_dir"], train_if_missing=False)
    with pytest.raises(ValueError):
        clf.classify("text", "word2vec", "svm", **kw)
    with pytest.raises(ValueError):
        clf.classify("text", "bow", "mlp", **kw)
    with pytest.raises(ValueError):
        clf.classify("text", "bow", "svm", cleaning="stem", **kw)
    with pytest.raises(ValueError):
        clf.classify("12345 !!! ???", "bow", "svm", **kw)  # pusty po czyszczeniu


def test_classify_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        clf.classify("space", "bow", "svm", models_dir=tmp_path / "m", train_if_missing=False)


def test_format_evaluation_and_summary(trained):
    d, results = trained
    text = clf.format_evaluation(results[("tfidf", "svm")])
    for token in ("ACC", "ERR", "macro avg", "weighted avg", "Macierz pomyłek"):
        assert token in text
    summary = clf.write_summary(d["results_dir"])
    assert len(summary) == 6
    assert (d["results_dir"] / "summary.csv").exists()
    assert (d["results_dir"] / "per_class_metrics.csv").exists()
    mats = json.loads((d["results_dir"] / "confusion_matrices.json").read_text())
    assert len(mats) == 6


def test_own_sentences_validation(trained):
    d, _ = trained
    rows = clf.validate_own_sentences(d["models_dir"], d["results_dir"])
    assert len(rows) == 6 * len(clf.OWN_SENTENCES)
    assert (d["results_dir"] / "own_sentences_predictions.csv").exists()
    assert {x["expected"] for x in rows} == set(cdata.CATEGORIES)


def test_run_all_writes_everything(tmp_path, split):
    d = {"models_dir": tmp_path / "m", "results_dir": tmp_path / "r"}
    summary = clf.run_all(split=split, **d)
    assert len(summary) == 6
    meta = json.loads((d["results_dir"] / "experiment_meta.json").read_text())
    assert meta["split"]["random_state"] == 42 and meta["categories"] == cdata.CATEGORIES
