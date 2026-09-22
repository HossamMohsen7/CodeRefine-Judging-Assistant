# Setup Guide

## Prerequisites (one-time)
- [ ] Python 3.11+ installed and on PATH (`python --version` shows a version)
- [ ] Poetry installed: `pip install poetry`
- [ ] Groq API key, console.groq.com then API Keys
- [ ] GitHub Personal Access Token, github.com then Settings, Developer settings, Personal access tokens, "repo" read scope
- [ ] A `SERVICE_AUTH_TOKEN` value agreed with whoever runs victoris-backend (any long random string works -- it's a shared secret, not a password anyone types in)

## Step 1, install dependencies
```bash
poetry install
```

## Step 2, set up your secrets
```bash
cp .env.example .env
```
Then fill in `.env` (no quotes, no spaces around `=`):
```
GROQ_API_KEY=your_real_key
GITHUB_TOKEN=your_real_token
SERVICE_AUTH_TOKEN=a_long_random_shared_secret
```

## Step 3, build the knowledge base (powers the chatbot)
```bash
poetry run python -m src.ingestion.build_knowledge_base
```
Reads `data/rules.md` (or `data/rules.pdf` if that exists instead), splits it into sections, embeds each with BGE-M3, saves to `data/chroma_db/`. First run downloads the BGE-M3 model (about 2GB, one time). Re-run whenever the rules document changes.

## Step 4, run the tests
```bash
poetry run pytest
```

## Step 5, run the API
```bash
poetry run uvicorn src.api.main:app --reload
```
Then `curl http://localhost:8000/health` should return `{"status":"ok"}`. `POST /chat` and `POST /grade` need `Authorization: Bearer <SERVICE_AUTH_TOKEN>`.

## Step 6, try the chatbot from the CLI (optional, local only)
```bash
poetry run python -m src.main chat
```

## Deploying it
```bash
docker compose up --build
```
See the Dockerfile at the project root. The port is configurable via a `PORT` environment variable.

## Common errors
- `ModuleNotFoundError`: you ran `python` instead of `poetry run python`
- `EnvironmentError: GROQ_API_KEY is not set` or similar: `.env` missing or not filled in
- `FileNotFoundError: data/rules.md`: not running from the project's root folder
- `401` on `/chat` or `/grade`: missing or wrong `Authorization: Bearer <SERVICE_AUTH_TOKEN>` header
- `422` on `/grade`: the repo URL couldn't be reached (private without `ieeemansb1` access, wrong URL, deleted repo, ...)
