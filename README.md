# Discord TODO + NLP bot

## Uruchomienie
Wymagana wersja Pythona: **3.11-3.13** (dla operacji `stem` z `pystempel`).

1. https://discord.com/developers/applications -> New Application -> Bot -> skopiuj token.
2. OAuth2 -> URL Generator: scopes `bot` + `applications.commands`, zaproś bota na serwer.
3. Instalacja (Windows, PowerShell):
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download pl_core_news_sm
4. `Copy-Item .env.example .env`, uzupełnij `DISCORD_TOKEN` i `ALBUM_LAST_DIGIT`.
5. `python bot.py`

## Publikacja na GitHub
1. Upewnij się, że używasz `.env` lokalnie (nie commituj sekretów).
2. Sprawdź status: `git status`.
3. Dodaj pliki: `git add .`.
4. Zrób pierwszy commit: `git commit -m "Initial commit"`.
5. Podłącz repozytorium i wypchnij:
   `git remote add origin <URL_REPO>`
   `git branch -M main`
   `git push -u origin main`

> Uwaga: jeśli token Discorda był gdziekolwiek upubliczniony, od razu go zresetuj w Discord Developer Portal i wpisz nowy do `.env`.

## Komendy
/tasks, /add, /update, /delete, /task <id> <operacja>
Operacje: tokens_nltk, tokens_spacy, tokens_compare, clean, normalize, unigram, bigram,
trigram, xgram, bow, tfidf, stem, lemma, stopwords
