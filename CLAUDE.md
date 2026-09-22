# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A stateless internal AI microservice for the CodeRefine system-design track (IEEE VICTORIS 4.0): an LLM agent grades a submitted GitHub repo against a fixed rubric, and a RAG chatbot answers rules questions. Called only by victoris-backend, which owns every team/judge/trial workflow, all persistence, and the participant/admin frontends -- this service has no logins, no database, and no memory of its own between calls. See `docs/specs/2026-09-22-coderefine-judging-platform.md` (in the sibling `coderefine-platform` repo) for the full cross-repo design; this repo implements only its "AI assistant (CodeRefine-Judging-Assistant) -- trimmed scope" section.

## Commands

Python 3.11+, Poetry. Groq-hosted LLMs.

```bash
poetry install
cp .env.example .env            # GROQ_API_KEY, GITHUB_TOKEN, SERVICE_AUTH_TOKEN
poetry run python -m src.ingestion.build_knowledge_base   # build Chroma KB from data/rules.md; needed before chat works
poetry run uvicorn src.api.main:app --reload               # POST /chat, POST /grade, GET /health
poetry run python -m src.main chat                          # CLI: local manual chatbot testing only
poetry run python -m src.visualize_graphs                   # render LangGraph diagrams to logs/*.png
poetry run pytest                                            # all tests
poetry run pytest tests/test_gather_node.py::test_gather_appends_committed_diagram_evidence   # single test
poetry run black src tests
docker compose up --build       # container listens on 7860, mapped to 5003
```

Tests monkeypatch module-level names (GitHub fetchers, `get_llm`, `_get_retriever`) rather than hitting the network. No linter besides black.

## Architecture

Two independent LangGraph systems that never call each other:

- **Grading agent** (`src/agent/`): `graph.py`'s `grade_repo(repo_url)` runs `gather -> format -> verify` (nodes in `nodes.py`) and returns the full graph state. `graph.py`'s `format_grade_response(final_scorecard, verification_notes)` reshapes that into POST /grade's exact `{score, feedback, evidence}` wire format. `verify` is plain code, no LLM: it checks every cited evidence ID exists in gather's output and flags invented rubric criteria. Evidence is cited **by ID, not by re-typed quote** -- don't reintroduce quote matching. `rubric.py` holds the fixed 5 criteria/weights; Bonus is judge-entered only in victoris-backend, never scored here. Team-submitted content is data, not instructions; prompt-injection attempts in READMEs get flagged in the evidence, not acted on. If `gather_node` can't reach the repo at all, it sets `repo_error` in the state, which `src/api/main.py` turns into a 422 instead of a fake scorecard.
- **Chatbot** (`src/chatbot/`, `src/ingestion/`): retrieve from Chroma (BGE-M3 embeddings) -> per-chunk LLM relevance check -> answer only from chunks that passed. `prompts.py` encodes the three-way rule: official facts only from the rules doc / general concepts labeled as general / real org procedures never guessed. **Stateless**: `graph.ask(question, history=None)` takes the full conversation history as a plain list of `{"role", "content"}` dicts on every call -- no `thread_id`, no checkpointer. The caller (victoris-backend) is the only place conversation history is stored.
- **Inputs** (`src/tools/`): GitHub repo reading, `.excalidraw` JSON parsing (no vision needed), and a vision model for raster/PDF diagrams (`diagram_reader.py`). SVG and external Figma/Excalidraw links are left to judges.
- **API** (`src/api/main.py`): FastAPI, three routes -- `POST /chat`, `POST /grade`, `GET /health`. `require_service_token` checks a single static bearer token (`SERVICE_AUTH_TOKEN` env var) on every route except `/health`; this service is internal-only, never called from a browser (no CORS, no rate limiting). `/grade` runs the grading graph synchronously, in-request -- no job queue (`ponytail:` fine at competition scale; if concurrent submissions become a problem, switch to a job id + webhook back to victoris-backend instead of blocking).
- **LLMs** (`src/agent/llm.py`): `get_llm()` is the only place clients are built (primary model with Llama fallback, cached by `(temperature, json_mode)`); use it, don't construct `ChatGroq` elsewhere.

## State and gotchas

- This service keeps no state of its own: no database, no session store, no on-disk audit log. `data/chroma_db/` (the knowledge-base vector store) is the only thing that needs to survive a restart, and is volume-mounted in `docker-compose.yml`.
- Rebuild the knowledge base after editing `data/rules.md`.
- `llm.py` fallback model id is `llama/llama-3.3-70b-versatile`; verify against Groq's actual id if fallback ever errors.
