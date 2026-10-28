"""Discord TODO bot z operacjami NLP (discord.py, komendy slash)."""
import asyncio
import os

import discord
from discord import app_commands
from dotenv import load_dotenv

import db
import nlp_utils

load_dotenv()
TOKEN = os.environ["DISCORD_TOKEN"]
GUILD_ID = os.getenv("GUILD_ID")  # opcjonalnie: natychmiastowa synchronizacja na jednym serwerze


class TodoBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()


bot = TodoBot()


# --- Pomocnicze --------------------------------------------------------------
def chunks(text: str, size: int = 1900):
    """Dzieli długi tekst na bloki kodu mieszczące się w limicie 2000 znaków Discorda."""
    lines, buf = text.split("\n"), ""
    for line in lines:
        while len(line) > size:
            if buf:
                yield buf
                buf = ""
            yield line[:size]
            line = line[size:]
        if len(buf) + len(line) + 1 > size:
            yield buf
            buf = ""
        buf += line + "\n"
    if buf.strip():
        yield buf


def fmt(t: db.Task) -> str:
    rel = f" ↳ `{t.relation[:8]}`" if t.relation else ""
    return f"`{t.id}` [{t.status}|{t.priority}] {t.description}{rel}"


async def task_autocomplete(_: discord.Interaction, current: str):
    cur = current.lower()
    return [
        app_commands.Choice(name=f"{t.id[:8]} – {t.description}"[:100], value=t.id)
        for t in db.list_tasks()
        if cur in t.id or cur in t.description.lower()
    ][:25]


# --- Komendy -----------------------------------------------------------------
@bot.tree.command(name="tasks", description="Wyświetl wszystkie zadania")
async def tasks(interaction: discord.Interaction):
    items = db.list_tasks()
    if not items:
        return await interaction.response.send_message("Brak zadań. Dodaj je przez `/add`.")
    text = "\n".join(fmt(t) for t in items)
    parts = list(chunks(text))
    await interaction.response.send_message(parts[0])
    for p in parts[1:]:
        await interaction.followup.send(p)


@bot.tree.command(name="add", description="Dodaj nowe zadanie")
@app_commands.describe(task="Treść zadania", priority="Priorytet", relation="ID zadania nadrzędnego")
@app_commands.choices(priority=[app_commands.Choice(name=p, value=p) for p in db.PRIORITIES])
@app_commands.autocomplete(relation=task_autocomplete)
async def add(interaction: discord.Interaction, task: str, priority: str = "normal", relation: str | None = None):
    try:
        t = db.add_task(task, priority, relation)
    except ValueError as e:
        return await interaction.response.send_message(f"❌ {e}", ephemeral=True)
    await interaction.response.send_message(f"Task created: `{t.id}`")


@bot.tree.command(name="update", description="Zaktualizuj istniejące zadanie")
@app_commands.describe(id="ID zadania", task="Nowa treść", status="Nowy status",
                       priority="Nowy priorytet", relation="Nowe zadanie nadrzędne")
@app_commands.choices(
    status=[app_commands.Choice(name=s, value=s) for s in db.STATUSES],
    priority=[app_commands.Choice(name=p, value=p) for p in db.PRIORITIES],
)
@app_commands.autocomplete(id=task_autocomplete, relation=task_autocomplete)
async def update(interaction: discord.Interaction, id: str, task: str | None = None,
                 status: str | None = None, priority: str | None = None, relation: str | None = None):
    try:
        t = db.update_task(id, task, status, priority, relation)
    except ValueError as e:
        return await interaction.response.send_message(f"❌ {e}", ephemeral=True)
    if not t:
        return await interaction.response.send_message("❌ Nie znaleziono zadania.", ephemeral=True)
    await interaction.response.send_message(f"Task updated: {fmt(t)}")


@bot.tree.command(name="delete", description="Usuń zadanie")
@app_commands.describe(id="ID zadania")
@app_commands.autocomplete(id=task_autocomplete)
async def delete(interaction: discord.Interaction, id: str):
    ok = db.delete_task(id)
    await interaction.response.send_message(
        f"Task deleted: `{id}`" if ok else "❌ Nie znaleziono zadania.", ephemeral=not ok
    )


@bot.tree.command(name="task", description="Pobierz zadanie i wykonaj operację NLP na jego opisie")
@app_commands.describe(id="ID zadania", field="Operacja NLP")
@app_commands.choices(field=[app_commands.Choice(name=o, value=o) for o in nlp_utils.OPERATIONS])
@app_commands.autocomplete(id=task_autocomplete)
async def task_cmd(interaction: discord.Interaction, id: str, field: str):
    t = db.get_task(id)
    if not t:
        return await interaction.response.send_message("❌ Nie znaleziono zadania.", ephemeral=True)
    await interaction.response.defer()  # modele NLP mogą potrzebować chwili
    corpus = [x.description for x in db.list_tasks()]
    try:
        result = await asyncio.to_thread(nlp_utils.run_operation, field, t.description, corpus)
    except Exception as e:  # np. brak modelu spaCy
        return await interaction.followup.send(f"❌ Błąd: {e}")
    await interaction.followup.send(f"{fmt(t)}\n**{field}**:")
    for part in chunks(result):
        await interaction.followup.send(f"```\n{part}```")


@bot.event
async def on_ready():
    print(f"Zalogowano jako {bot.user} | x-gram = {nlp_utils.X}")


if __name__ == "__main__":
    db.init_db()
    bot.run(TOKEN)
