import random

import pytest

import classification as clf
import classification_data as cdata
import db

VOCAB = {
    "comp.graphics": "image pixel render polygon texture color vector display graphics bitmap shader screen "
                     "software video drawing animation".split(),
    "sci.space": "orbit rocket planet satellite shuttle launch moon mars astronaut telescope nasa solar "
                 "galaxy spacecraft universe lunar".split(),
    "rec.sport.baseball": "pitcher inning homerun bat team season batter league playoff baseman game "
                          "stadium catcher score coach".split(),
    "talk.politics.misc": "senator election president government congress law tax vote policy party "
                          "court budget campaign citizen debate".split(),
}
COMMON = "the and with about really people think good new".split()


def synthetic_corpus(per_class=40, seed=0):
    """Mały syntetyczny angielski korpus (TYLKO do testów logiki, nie są to wyniki eksperymentów)."""
    rng = random.Random(seed)
    texts, labels = [], []
    for name, words in VOCAB.items():
        for _ in range(per_class):
            toks = [rng.choice(words) for _ in range(14)] + [rng.choice(COMMON) for _ in range(4)]
            rng.shuffle(toks)
            texts.append(" ".join(toks))
            labels.append(name)
    return texts, labels


@pytest.fixture(scope="session")
def split():
    texts, labels = synthetic_corpus()
    return cdata.make_split(texts, labels)


@pytest.fixture()
def dirs(tmp_path):
    return {"models_dir": tmp_path / "models", "results_dir": tmp_path / "results"}


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "tasks_test.db"))
    db.init_db()
    return db
