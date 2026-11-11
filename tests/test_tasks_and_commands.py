import os
import sqlite3

import pytest

import classification_commands as cc
import db


def test_migration_adds_label_and_keeps_rows(tmp_path, monkeypatch):
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE tasks (id TEXT PRIMARY KEY, description TEXT NOT NULL,
        relation TEXT, status TEXT NOT NULL DEFAULT 'todo', priority TEXT NOT NULL DEFAULT 'normal',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    con.execute("INSERT INTO tasks VALUES ('id-1','Stare zadanie',NULL,'todo','normal','2024','2024')")
    con.commit(); con.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db(); db.init_db()  # idempotentne
    t = db.get_task("id-1")
    assert t.description == "Stare zadanie" and t.label is None
    assert db.add_task("Nowe").label is None  # INSERT działa na nowym schemacie


def test_set_label_and_overwrite(temp_db):
    t = temp_db.add_task("Napisać raport")
    assert temp_db.set_label(t.id, "praca").label == "praca"
    assert temp_db.set_label(t.id, "nauka").label == "nauka"  # nadpisanie
    assert temp_db.get_task(t.id).label == "nauka"
    assert temp_db.set_label("brak", "dom") is None
    with pytest.raises(ValueError):
        temp_db.set_label(t.id, "hobby")
    assert temp_db.get_task(t.id).label == "nauka"  # błędna etykieta nic nie zmienia


def test_unlabeled_and_no_data_loss(temp_db):
    a, b = temp_db.add_task("A"), temp_db.add_task("B")
    temp_db.set_label(a.id, "dom")
    assert [t.id for t in temp_db.list_tasks(unlabeled_only=True)] == [b.id]
    assert len(temp_db.list_tasks()) == 2


def test_label_messages(temp_db):
    t = temp_db.add_task("Kupić mleko")
    assert "Kupić mleko" in cc.unlabeled_message()
    assert "Label saved" in cc.assign_label_message(t.id, "dom")
    assert "poprzednio: dom" in cc.assign_label_message(t.id, "praca")
    assert "Wszystkie zadania mają etykiety" in cc.unlabeled_message()
    with pytest.raises(ValueError):
        cc.assign_label_message(t.id, "xyz")
    with pytest.raises(ValueError):
        cc.assign_label_message("nie-ma", "dom")


def test_classifier_message_validation():
    with pytest.raises(ValueError):
        cc.classifier_message("   ", "bow", "svm", "clean")
    with pytest.raises(ValueError):
        cc.classifier_message("x" * (cc.MAX_TEXT_LEN + 1), "bow", "svm", "clean")
    with pytest.raises(ValueError):
        cc.classifier_message("space", "bow", "mlp", "clean")
    with pytest.raises(ValueError):
        cc.evaluate_message("bow", "mlp")


def test_existing_commands_preserved(monkeypatch):
    """Zachowanie funkcji z zadania 1: stare komendy nadal zarejestrowane + 3 nowe."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    import bot

    names = {c.name for c in bot.bot.tree.get_commands()}
    assert {"tasks", "add", "update", "delete", "task"} <= names
    assert {"prepare_data_for_classification", "classifier", "evaluate"} <= names
    cls = next(c for c in bot.bot.tree.get_commands() if c.name == "classifier")
    required = [p.name for p in cls.parameters if p.required]
    assert required == ["cleaning", "text", "method_repr", "method_class"]


def test_legacy_nlp_functions(monkeypatch):
    import importlib

    import nlp_utils

    monkeypatch.setenv("ALBUM_LAST_DIGIT", "0")  # ostatnia cyfra albumu 0 -> x = 0 + 4 = 4
    nlp_utils = importlib.reload(nlp_utils)
    assert nlp_utils.X == 4
    assert nlp_utils.clean_text("Hello, Świat 123!") == "hello świat"
    text = "Przygotować prezentację z przetwarzania języka naturalnego"
    assert len(nlp_utils.make_ngrams(text, 2)) == 4
    assert "przygotować" in nlp_utils.run_operation("bigram", text, [text])
    assert "TfidfVectorizer" in nlp_utils.run_operation("tfidf", text, [text, "inne zadanie"])


def test_auto_train_once_then_cached(tmp_path, monkeypatch, split):
    """/classifier i /evaluate bez modelu: trenują raz (dane syntetyczne), potem czytają zapis."""
    import classification as clf

    calls = {"n": 0}

    def fake_split():
        calls["n"] += 1
        return split

    monkeypatch.setattr(clf, "get_split", fake_split)
    monkeypatch.setattr(clf, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(clf, "RESULTS_DIR", tmp_path / "results")

    first = cc.classifier_message("Rocket launch to the moon", "tfidf", "logistic_regression", "clean")
    assert "Predicted class" in first and "p=" in first and "wytrenowano" in first
    second = cc.classifier_message("Rocket launch to the moon", "tfidf", "logistic_regression", "clean")
    assert "wytrenowano" not in second and calls["n"] == 1  # brak ponownego trenowania

    svm = cc.classifier_message("Rocket launch to the moon", "bow", "svm", "clean")
    assert "p=" not in svm  # SVM bez prawdopodobieństwa

    text, trained = cc.evaluate_message("tfidf", "logistic_regression")
    assert not trained and "ACC" in text            # wynik z zapisu
    text, trained = cc.evaluate_message("bow", "svm")
    assert not trained                              # wyniki zapisał już /classifier
    text, trained = cc.evaluate_message("embedding", "svm")
    assert trained and "Macierz pomyłek" in text    # brak wyników -> policzone, nie zmyślone
