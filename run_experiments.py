"""Uruchamia 6 eksperymentów (BoW/TF-IDF/embedding x Linear SVM/Logistic Regression) na 20 Newsgroups.

Użycie:
    python run_experiments.py           # trenuje brakujące modele, zapisuje wyniki w results/
    python run_experiments.py --force   # trenuje od nowa wszystkie 6 konfiguracji
Wymaga internetu przy pierwszym uruchomieniu (fetch_20newsgroups) i modelu en_core_web_md.
"""
import argparse
import logging

import classification as clf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="wytrenuj modele od nowa")
    ap.add_argument("--no-own-sentences", action="store_true", help="pomiń walidację na własnych zdaniach")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = clf.run_all(force=args.force, own_sentences=not args.no_own_sentences)
    print("\n" + clf.format_summary(summary))
    print(f"\nWyniki zapisano w: {clf.RESULTS_DIR.resolve()}")
    print(f"Modele zapisano w: {clf.MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
