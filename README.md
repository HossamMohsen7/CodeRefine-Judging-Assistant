# CodeRefine Judging Assistant

A stateless internal AI microservice for CodeRefine's system-design track: an LLM agent grades a submitted GitHub repo against the real judging rubric, and a separate RAG chatbot answers teams' questions about the rules. It has no logins, no database, and no memory between calls -- victoris-backend owns every team/judge workflow and all persistence, and calls this service over HTTP with a shared bearer token. See `CLAUDE.md` for commands and architecture.

## API surface

```
POST /grade { repoUrl: string } -> { score: number, feedback: string, evidence: string }
                                 -> 422 { error: string }   (repo unreachable/unreadable)

POST /chat { question: string, history?: { role: "user" | "assistant", content: string }[] }
          -> { answer: string }

GET  /health -> { status: "ok" }
```

`POST /chat` and `POST /grade` require `Authorization: Bearer <SERVICE_AUTH_TOKEN>`; `GET /health` doesn't.

## How grading works

```
POST /grade { repoUrl }
        v
 gather  -- reads the repo, produces factual, file-cited observations
        v
 format  -- turns observations into a scored draft against rubric.py
        v
 verify  -- plain-code check: every cited evidence id actually exists
        v
{ score, feedback, evidence }
```

Nothing here auto-publishes a result to a team: this response is an AI-drafted score that victoris-backend stores and a judge reviews before it's ever released.

## How the chatbot works

```
POST /chat { question, history }
        v
 retrieve candidate rule sections for `question` (Chroma / BGE-M3)
        v
 grade each one for genuine relevance (separate LLM call)
        v
 answer using only sections that passed, plus the full supplied `history`
        v
{ answer }
```

No conversation state is kept here -- `history` is exactly what the caller wants remembered, on every call.

## Folder map

| Folder | What's in it |
|---|---|
| `src/agent/` | The grading agent: rubric, LLM setup, the gather/format/verify pipeline, and the response formatter |
| `src/tools/` | Reading a team's GitHub repo and parsing the architecture diagram |
| `src/ingestion/` | Turns the rules document into a searchable knowledge base for the chatbot |
| `src/chatbot/` | The stateless team support chatbot |
| `src/api/` | The FastAPI app: bearer-token auth, `/chat`, `/grade`, `/health` |
| `data/` | The rules document (and the generated knowledge base, once built) |

Each `src/` subfolder has its own README with the file-by-file detail.
