# Discord TODO + NLP bot

## Uruchomienie
1. https://discord.com/developers/applications -> New Application -> Bot -> skopiuj token.
2. OAuth2 -> URL Generator: scopes `bot` + `applications.commands`, zaproś bota na serwer.
3. Instalacja:
   pip install -r requirements.txt
   python -m spacy download pl_core_news_sm
4. `cp .env.example .env`, uzupełnij `DISCORD_TOKEN` i `ALBUM_LAST_DIGIT`.
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
