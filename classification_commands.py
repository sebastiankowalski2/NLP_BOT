"""Komendy Discorda zadania 2: /prepare_data_for_classification, /classifier, /evaluate.

Logika (funkcje *_message) jest oddzielona od handlerów Discorda, dzięki czemu da się ją testować.
Rejestracja: classification_commands.setup(bot.tree)  (jedna linia w bot.py).
"""
import asyncio

import discord
from discord import app_commands

import classification as clf
import classification_data as cdata
import db

MAX_TEXT_LEN = 5000


# --- Logika komend (zwraca tekst odpowiedzi; rzuca ValueError przy złych argumentach) ------
def assign_label_message(task_id: str, label: str) -> str:
    previous = db.get_task(task_id)
    t = db.set_label(task_id, label)  # ValueError przy złej etykiecie
    if t is None:
        raise ValueError("Nie znaleziono zadania.")
    old = f" (poprzednio: {previous.label})" if previous and previous.label else ""
    labeled = sum(1 for x in db.list_tasks() if x.label)
    return (f"Label saved: **{label}**{old} → `{t.id[:8]}` {t.description}\n"
            f"Oznaczone zadania: {labeled}/{len(db.list_tasks())}. Klasy: {', '.join(db.LABELS)}")


def unlabeled_message() -> str:
    tasks = db.list_tasks(unlabeled_only=True)
    if not tasks:
        return "Wszystkie zadania mają etykiety (albo lista zadań jest pusta)."
    lines = [f"Zadania bez etykiety ({len(tasks)}):"] + [f"`{t.id[:8]}` {t.description}" for t in tasks]
    lines.append("\nPrzypisz: `/prepare_data_for_classification task:<zadanie> label:<praca|dom|nauka>`")
    return "\n".join(lines)


def classifier_message(text: str, method_repr: str, method_class: str, cleaning: str) -> str:
    if not text or not text.strip():
        raise ValueError("Podaj tekst do sklasyfikowania.")
    if len(text) > MAX_TEXT_LEN:
        raise ValueError(f"Tekst jest za długi (max {MAX_TEXT_LEN} znaków).")
    r = clf.classify(text, method_repr, method_class, cleaning)
    msg = clf.format_classification(r)
    if r["trained_now"]:
        msg += "\n(model nie był jeszcze wytrenowany - wytrenowano go teraz i zapisano na dysku)"
    return msg + "\nModel wytrenowany na 20 Newsgroups (angielski): " + ", ".join(cdata.CATEGORIES)


def evaluate_message(method_repr: str, method_class: str) -> tuple[str, bool]:
    """Zwraca (wynik, czy_trenowano_teraz). Brakujące wyniki są liczone, nigdy zmyślane."""
    clf.validate_config(method_repr, method_class)
    existing = clf.load_result(method_repr, method_class)
    result = existing or clf.train_and_evaluate(method_repr, method_class)
    return clf.format_evaluation(result), existing is None


# --- Pomocnicze Discorda -----------------------------------------------------------------
def _chunks(text: str, size: int = 1900):
    buf = ""
    for line in text.split("\n"):
        line = line[:size]
        if len(buf) + len(line) + 1 > size:
            yield buf
            buf = ""
        buf += line + "\n"
    if buf.strip():
        yield buf


async def _task_label_autocomplete(_: discord.Interaction, current: str):
    cur = current.lower()
    tasks = sorted(db.list_tasks(), key=lambda t: t.label is not None)  # bez etykiet na górze
    return [
        app_commands.Choice(name=f"[{t.label or 'brak'}] {t.id[:8]} – {t.description}"[:100], value=t.id)
        for t in tasks if cur in t.id or cur in t.description.lower()
    ][:25]


def setup(tree: app_commands.CommandTree) -> None:
    @tree.command(name="prepare_data_for_classification",
                  description="Przypisz etykietę praca/dom/nauka do zadania (bez argumentów: lista bez etykiet)")
    @app_commands.describe(task="Zadanie do oznaczenia", label="Etykieta (nadpisuje poprzednią)")
    @app_commands.choices(label=[app_commands.Choice(name=l, value=l) for l in db.LABELS])
    @app_commands.autocomplete(task=_task_label_autocomplete)
    async def prepare_data(interaction: discord.Interaction, task: str | None = None,
                           label: str | None = None):
        if task is None and label is None:
            parts = list(_chunks(unlabeled_message()))
            await interaction.response.send_message(parts[0])
            for p in parts[1:]:
                await interaction.followup.send(p)
            return
        if task is None or label is None:
            return await interaction.response.send_message(
                "❌ Podaj oba argumenty: `task` i `label` (albo żadnego, aby zobaczyć zadania bez etykiety).",
                ephemeral=True)
        try:
            msg = assign_label_message(task, label)
        except ValueError as e:
            return await interaction.response.send_message(f"❌ {e}", ephemeral=True)
        await interaction.response.send_message(msg)

    @tree.command(name="classifier", description="Sklasyfikuj angielski tekst modelem wytrenowanym na 20 Newsgroups")
    @app_commands.describe(
        cleaning="Czyszczenie tekstu przed klasyfikacją (wymagane)",
        text="Tekst do sklasyfikowania (po angielsku)",
        method_repr="Reprezentacja tekstu", method_class="Klasyfikator")
    @app_commands.choices(
        cleaning=[app_commands.Choice(name=c, value=c) for c in cdata.CLEANING_METHODS],
        method_repr=[app_commands.Choice(name=r, value=r) for r in clf.REPRESENTATIONS],
        method_class=[app_commands.Choice(name=c, value=c) for c in clf.CLASSIFIERS])
    async def classifier(interaction: discord.Interaction, cleaning: str, text: str,
                         method_repr: str, method_class: str):
        await interaction.response.defer()  # ewentualne trenowanie może chwilę potrwać
        try:
            msg = await asyncio.to_thread(classifier_message, text, method_repr, method_class, cleaning)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            return await interaction.followup.send(f"❌ {e}")
        except Exception as e:  # np. brak internetu przy pierwszym pobraniu 20 Newsgroups
            return await interaction.followup.send(f"❌ Błąd klasyfikacji: {type(e).__name__}: {e}")
        await interaction.followup.send(msg)

    @tree.command(name="evaluate", description="Macierz pomyłek i metryki dla wybranej konfiguracji (20 Newsgroups)")
    @app_commands.describe(method_repr="Reprezentacja tekstu", method_class="Klasyfikator")
    @app_commands.choices(
        method_repr=[app_commands.Choice(name=r, value=r) for r in clf.REPRESENTATIONS],
        method_class=[app_commands.Choice(name=c, value=c) for c in clf.CLASSIFIERS])
    async def evaluate(interaction: discord.Interaction, method_repr: str, method_class: str):
        await interaction.response.defer()
        try:
            text, trained = await asyncio.to_thread(evaluate_message, method_repr, method_class)
        except (ValueError, RuntimeError) as e:
            return await interaction.followup.send(f"❌ {e}")
        except Exception as e:
            return await interaction.followup.send(f"❌ Błąd ewaluacji: {type(e).__name__}: {e}")
        if trained:
            await interaction.followup.send("Brak zapisanych wyników - wytrenowano model i obliczono metryki.")
        for part in _chunks(text):
            await interaction.followup.send(f"```\n{part}```")
