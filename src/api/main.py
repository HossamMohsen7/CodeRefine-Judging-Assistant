"""
Stateless internal API: POST /chat (RAG rules chatbot), POST /grade (repo
grading), GET /health. Called only by victoris-backend, authenticated with
a single static bearer token -- see
docs/specs/2026-09-22-coderefine-judging-platform.md, "AI assistant
(CodeRefine-Judging-Assistant) -- trimmed scope".
"""

from dotenv import load_dotenv
load_dotenv()

import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.agent.graph import format_grade_response, grade_repo
from src.chatbot.chatbot import answer_question

app = FastAPI(title="CodeRefine AI Assistant")

_bearer_scheme = HTTPBearer(auto_error=False)


def require_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """
    Single static bearer token, checked on every route except /health --
    this service is internal-only, never called directly by a browser.
    """
    expected = os.environ.get("SERVICE_AUTH_TOKEN")
    if not expected or credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] | None = None


class ChatResponse(BaseModel):
    answer: str


class GradeRequest(BaseModel):
    repoUrl: str


class GradeResponse(BaseModel):
    score: int
    feedback: str
    evidence: str


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_service_token)])
def chat(req: ChatRequest) -> dict:
    history = [turn.model_dump() for turn in req.history] if req.history else None
    return {"answer": answer_question(req.question, history=history)}


@app.post("/grade", response_model=GradeResponse, dependencies=[Depends(require_service_token)])
def grade(req: GradeRequest):
    # ponytail: synchronous call, no job queue -- fine at competition scale
    # (a handful of teams submitting a handful of times). If grading
    # latency or concurrent submissions become a problem, switch to
    # POST /grade returning a job id + a webhook back to victoris-backend
    # instead of blocking the request. (spec's own note, same wording)
    result = grade_repo(repo_url=req.repoUrl)
    if result.get("repo_error"):
        # The contract's 422 body is {"error": string} -- raising
        # HTTPException(detail=...) here would wrap it as {"detail": ...}
        # instead, so this returns the JSON body directly.
        return JSONResponse(status_code=422, content={"error": result["repo_error"]})
    return format_grade_response(result["final_scorecard"], result["verification_notes"])


@app.get("/health")
def health() -> dict:
    """No auth needed -- confirms the API is running; orchestrators/load balancers hit this."""
    return {"status": "ok"}
