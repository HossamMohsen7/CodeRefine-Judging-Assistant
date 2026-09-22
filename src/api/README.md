# src/api/

`main.py` is the whole API: a bearer-token dependency plus three routes. No grading or judging logic lives here -- each route just calls a function that already exists in `src/agent/` or `src/chatbot/`.

| Endpoint | Calls | Auth |
|---|---|---|
| `POST /chat` | `src/chatbot/chatbot.py`'s `answer_question(question, history)` | `Authorization: Bearer <SERVICE_AUTH_TOKEN>` |
| `POST /grade` | `src/agent/graph.py`'s `grade_repo(repo_url)` + `format_grade_response(...)` | `Authorization: Bearer <SERVICE_AUTH_TOKEN>` |
| `GET /health` | nothing, just returns `{"status": "ok"}` | none |

`/grade` runs the grading graph synchronously, in the request -- no background job queue. `ponytail:` fine at competition scale (a handful of teams submitting a handful of times); if grading latency or concurrent submissions become a problem, switch to returning a job id + a webhook back to victoris-backend instead.

`/grade` returns `422 {"error": "..."}` when the repo itself couldn't be reached at all (bad URL, private without access, deleted). Any other failure mode (thin README, no architecture diagram, etc.) still returns `200` with a low-confidence scorecard -- a judge reviews every AI draft in victoris-dashboard before it's released to a team.

## Running it

```bash
poetry run uvicorn src.api.main:app --reload
```
