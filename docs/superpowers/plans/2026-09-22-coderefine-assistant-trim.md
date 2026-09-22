# CodeRefine-Judging-Assistant Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trim CodeRefine-Judging-Assistant from a stateful team/judge web app (passwords, sessions, SQLite review queue, practice-trial pipeline, background job queue, static website) down to a stateless internal AI microservice exposing exactly `POST /chat`, `POST /grade`, `GET /health`, authenticated by a single static bearer token, called only by victoris-backend.

**Architecture:** Keep the existing `gather → format → verify` LangGraph grading pipeline (`src/agent/`), the RAG chatbot graph (`src/chatbot/`), the knowledge-base ingestion (`src/ingestion/`), and the GitHub/diagram tools (`src/tools/`) exactly as they already work. Delete every module whose job is persistence, session auth, attempt counting, or judge workflow (that responsibility now lives entirely in victoris-backend). Rewrite `src/api/main.py` as a thin, stateless FastAPI shell: a bearer-token dependency, and three routes that call straight into the kept pipelines. The grading pipeline gains one new pure function, `format_grade_response()`, that reshapes its internal scorecard into the exact `{score, feedback, evidence}` wire format. The chatbot graph drops its `InMemorySaver`/`thread_id` checkpoint in favor of a caller-supplied `history` list, since the assistant itself must keep no conversation state.

**Tech Stack:** Python 3.11, Poetry, FastAPI + Uvicorn, LangGraph, LangChain (Groq LLMs, Chroma vector store, HuggingFace embeddings), PyGithub, pytest (existing convention: plain functions + `monkeypatch`, no fixtures/conftest.py).

**Spec:** `e:\Developing\IEEE\coderefine-platform\docs\specs\2026-09-22-coderefine-judging-platform.md` — this plan implements only the "AI assistant (CodeRefine-Judging-Assistant) — trimmed scope" section and the matching "CodeRefine-Judging-Assistant (internal only, `Authorization: Bearer <static token>`)" API contract block. victoris-backend and coderefine-platform are owned by other plans/agents and are out of scope here.

## Global Constraints

- Package manager: Poetry (this repo's own convention — `poetry install`, `poetry run ...`; unrelated to any other repo's pnpm/npm rules).
- Keep exactly: `src/chatbot/` (RAG chat graph), `src/ingestion/` (knowledge-base build/retriever), `src/agent/` grading graph (`gather → format → verify`), `src/tools/` (github_tool, diagram parsers), `src/agent/llm.py`.
- Remove entirely: `src/api/auth.py` (login/sessions), `attempt_tracker.py`, `review_queue.py`, `practice_store.py`, `batch_grade.py`, all SQLite persistence (`logs/*.db`), the website (`web/`, which only served the old team/judge login UI), `logs/questions.jsonl` / `logs/scorecards.jsonl` audit logging — victoris-backend now owns the record of every trial and message.
- New stateless API surface, exact shapes (binding, from the spec's API contracts section):
  - `POST /chat { question: string, history?: { role: "user" | "assistant", content: string }[] } → 200 { answer: string }` — no `thread_id`, no `InMemorySaver` checkpoint; caller supplies whatever context it wants remembered.
  - `POST /grade { repoUrl: string } → 200 { score: number, feedback: string, evidence: string }`, `422 { error: string }` on repo unreachable/unreadable.
  - `GET /health → 200 { status: "ok" }`.
  - Auth: a single static bearer token (env var), checked on every route except `/health` — this service is never called directly by a browser.
- `ponytail:` `/grade` is synchronous by design for now — no job queue; fine at competition scale (a handful of teams submitting a handful of times). If grading latency or concurrent submissions become a problem, switch to `POST /grade` returning a job id + a webhook back to victoris-backend instead of blocking the request. (This is the spec's own note, carried into code as a matching comment in Task 3.)
- Every request/response field name must exactly match the spec's API contracts section (`repoUrl`, `question`, `history`, `answer`, `score`, `feedback`, `evidence`, `error`, `status`) — no renaming, no extra/missing fields.

---

## Task 1: Trim the grading graph to drop the practice-trial pipeline and produce `/grade`'s exact response shape

**Files:**
- Modify: `src/agent/state.py`
- Modify: `src/agent/nodes.py`
- Modify: `src/agent/graph.py`
- Modify: `src/visualize_graphs.py`
- Test: `tests/test_gather_node.py` (add one test)
- Test: `tests/test_grade_formatting.py` (new)

**Interfaces:**
- Consumes: nothing new (all inputs are already in the repo).
- Produces:
  - `src.agent.state.GradingState` — TypedDict with keys `repo_url, file_tree, readme_content, raw_notes, draft_scorecard, final_scorecard, verification_notes, repo_error` (no `team_name`, no `PracticeFeedbackState`/`CriterionFeedback`).
  - `src.agent.graph.grade_repo(repo_url: str) -> dict` — runs `gather → format → verify`, returns the full final `GradingState` dict.
  - `src.agent.graph.format_grade_response(final_scorecard: list[dict], verification_notes: str) -> dict` — returns `{"score": int, "feedback": str, "evidence": str}`.
  - `nodes.gather_node`'s except-branch now also sets `"repo_error": str(e)` in its returned dict (all other keys unchanged) so callers can tell "repo unreachable" apart from "repo reachable but evidence is thin".

Every trial in the new platform (1st, 2nd, 3rd) goes through the same AI-drafted-then-judge-reviewed cycle (see the spec's "Trial flow" section) — there is no more attempt-count-based routing to a separate no-score "practice" pipeline. That routing lived in `attempt_tracker.py` + `graph.submit_attempt()` + `nodes.feedback_node()` + `state.PracticeFeedbackState`, all of which become dead code once `/grade` always runs the official `gather → format → verify` pipeline. `team_name` also drops out of `GradingState`: grep confirms no node ever reads `state["team_name"]` for grading logic (`tests/test_gather_node.py` already calls `nodes.gather_node({"repo_url": ...})` with no `team_name` key at all), and the new `/grade` contract doesn't send one.

- [x] **Step 1: Write the failing test for the new `repo_error` field**

Add to `tests/test_gather_node.py` (after the existing `test_gather_appends_committed_diagram_evidence` test):

```python
def test_gather_flags_repo_error_when_repo_unreachable(monkeypatch):
    def _raise(_repo_url):
        raise RuntimeError("Could not access repo 'owner/repo': 404")

    monkeypatch.setattr(nodes, "find_submission_files", _raise)

    result = nodes.gather_node({"repo_url": "https://github.com/owner/repo"})

    assert result["repo_error"] == "Could not access repo 'owner/repo': 404"
    assert "Could not access repo" in result["raw_notes"][0]["observation"]
```

- [x] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/test_gather_node.py::test_gather_flags_repo_error_when_repo_unreachable -v`
Expected: FAIL with `KeyError: 'repo_error'`.

- [x] **Step 3: Add `repo_error` to `gather_node`'s except branch**

In `src/agent/nodes.py`, `gather_node`'s `try/except` around `find_submission_files(repo_url)` currently returns:

```python
    try:
        found_files = find_submission_files(repo_url)
    except Exception as e:
        # Can't access the repo at all
        # flagged low-confidence downstream. We still continue so the
        # agent produces *something* the judge can see, rather than crashing.
        return {
            "file_tree": [],
            "readme_content": "",
            "raw_notes": [
                {
                    "file_path": "N/A",
                    "line_range": "N/A",
                    "excerpt": "",
                    "observation": f"Could not access repo: {e}",
                    "source_type": "text",
                }
            ],
        }
```

Change the returned dict to also carry `repo_error`, so `/grade` can distinguish "couldn't reach the repo at all" (422) from "reached it, evidence is just thin" (a normal, low-confidence scorecard):

```python
    try:
        found_files = find_submission_files(repo_url)
    except Exception as e:
        # Can't access the repo at all -- flagged low-confidence downstream
        # AND surfaced as repo_error, which POST /grade turns into a 422
        # instead of a fake scorecard (see src/api/main.py).
        return {
            "file_tree": [],
            "readme_content": "",
            "raw_notes": [
                {
                    "file_path": "N/A",
                    "line_range": "N/A",
                    "excerpt": "",
                    "observation": f"Could not access repo: {e}",
                    "source_type": "text",
                }
            ],
            "repo_error": str(e),
        }
```

Leave every other line of `gather_node`, `format_node`, and `verify_node` unchanged.

- [x] **Step 4: Delete `feedback_node`**

Still in `src/agent/nodes.py`, delete the entire `feedback_node` function (the practice-trial equivalent of `format_node`, at the end of the file — the docstring starts with `"""The practice-trial equivalent of format_node..."""`). Nothing else in the kept pipeline calls it.

- [x] **Step 5: Run the gather tests to verify they pass**

Run: `poetry run pytest tests/test_gather_node.py -v`
Expected: PASS (both `test_gather_appends_committed_diagram_evidence` and the new `test_gather_flags_repo_error_when_repo_unreachable`).

- [x] **Step 6: Trim `src/agent/state.py`**

Replace the full contents of `src/agent/state.py` with:

```python
"""
The shared state that flows through every step of the grading agent
(gather -> format -> verify). Think of this as the agent's "memory" for
one grading run: each step reads from it and adds to it.
"""

from typing import Optional
from typing_extensions import TypedDict

class CriterionEvidence(TypedDict):
    """One piece of evidence the agent found while inspecting the repo."""
    id: int            # stable numeric id, used by format_node to cite this note
    file_path: str
    line_range: str   # "22-45"
    excerpt: str      # short excerpt supporting the observation
    observation: str  # what this evidence shows, in plain language
    source_type: str  # "text" | "excalidraw" | "image" | "pdf"


class CriterionScorecard(TypedDict):
    """The final, judge-facing result for a single rubric criterion."""
    criterion: str            # like : "Data Model"
    score_percent: int        # 0 to that criterion's weight_percent, see rubric.py
    justification: str        # short written reasoning
    evidence: list[CriterionEvidence]  # resolved by verify_node from cited evidence_ids
    confidence: str           # "high" | "medium" | "low"  per-criterion, not global


class GradingState(TypedDict):
    """
    The full state object LangGraph passes between nodes for one grading
    run. No team_name: nothing in gather/format/verify reads it, and
    POST /grade's request body doesn't carry one (victoris-backend tracks
    which team a repo belongs to, not this service).
    """
    # --- input, set once at the start ---
    repo_url: str

    # --- filled in by the "gather" step ---
    file_tree: Optional[list[str]]
    readme_content: Optional[str]
    raw_notes: Optional[list[CriterionEvidence]]  # ungraded observations, tied to files/lines
    repo_error: Optional[str]  # set only when the repo itself couldn't be reached at all

    # --- filled in by the "format" step ---
    draft_scorecard: Optional[list[CriterionScorecard]]

    # --- filled in by the "verify" step ---
    final_scorecard: Optional[list[CriterionScorecard]]
    verification_notes: Optional[str]  # any mismatches found and how they were handled
```

- [x] **Step 7: Write the failing test for `format_grade_response`**

Create `tests/test_grade_formatting.py`:

```python
from src.agent.graph import format_grade_response


def test_format_grade_response_sums_score_and_includes_criteria():
    scorecard = [
        {
            "criterion": "Data Model",
            "score_percent": 15,
            "justification": "Covers all entities.",
            "confidence": "high",
            "evidence": [{"file_path": "README.md", "line_range": "10-20", "observation": "Entities listed."}],
        },
        {
            "criterion": "API Design",
            "score_percent": 10,
            "justification": "Partial coverage.",
            "confidence": "low",
            "evidence": [],
        },
    ]

    result = format_grade_response(scorecard, "All evidence checked out.")

    assert result["score"] == 25
    assert "Data Model" in result["feedback"]
    assert "API Design" in result["feedback"]
    assert "README.md" in result["evidence"]
    assert "No cited evidence." in result["evidence"]
    assert result["evidence"].startswith("Verification: All evidence checked out.")
```

- [x] **Step 8: Run it to verify it fails**

Run: `poetry run pytest tests/test_grade_formatting.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_grade_response'`.

- [x] **Step 9: Rewrite `src/agent/graph.py`**

Replace the full contents of `src/agent/graph.py` with:

```python
"""
Wires gather -> format -> verify into one runnable LangGraph flow, and
turns the resulting scorecard into POST /grade's exact response shape
(see docs/specs/2026-09-22-coderefine-judging-platform.md, API contracts).

grade_repo() only describes the ORDER things happen in.
"""

from langgraph.graph import StateGraph, END

from src.agent.state import GradingState
from src.agent.nodes import gather_node, format_node, verify_node


def build_grading_graph():
    graph = StateGraph(GradingState)

    graph.add_node("gather", gather_node)
    graph.add_node("format", format_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("gather")
    graph.add_edge("gather", "format")
    graph.add_edge("format", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def grade_repo(repo_url: str) -> dict:
    """
    The one function POST /grade calls. Returns the full final
    GradingState dict, including final_scorecard, verification_notes, and
    repo_error (set only when the repo itself couldn't be reached -- see
    gather_node's except branch in nodes.py).
    """
    app = build_grading_graph()
    initial_state: GradingState = {
        "repo_url": repo_url,
        "file_tree": None,
        "readme_content": None,
        "raw_notes": None,
        "repo_error": None,
        "draft_scorecard": None,
        "final_scorecard": None,
        "verification_notes": None,
    }
    return app.invoke(initial_state)


def format_grade_response(final_scorecard: list[dict], verification_notes: str) -> dict:
    """
    Turns the grading graph's internal final_scorecard + verification_notes
    into POST /grade's exact {score, feedback, evidence} response shape.

    score: the sum of each criterion's score_percent (0-100; rubric.py's
    weights sum to 100). Bonus is judge-only and never appears here.
    feedback: one readable line per criterion (score, confidence,
    justification) -- what a judge reads to sanity-check the AI draft.
    evidence: the citation trail behind that feedback (verification notes,
    then each criterion's cited file/line/observation), kept as a
    separate field so feedback stays skimmable.
    """
    score = sum(item.get("score_percent", 0) for item in final_scorecard)

    feedback_lines = []
    evidence_lines = [f"Verification: {verification_notes}", ""]
    for item in final_scorecard:
        feedback_lines.append(
            f"{item['criterion']} ({item.get('score_percent', 0)}%, "
            f"{item.get('confidence', 'low')} confidence): "
            f"{item.get('justification', 'No justification recorded.')}"
        )
        evidence_lines.append(f"## {item['criterion']}")
        criterion_evidence = item.get("evidence") or []
        if criterion_evidence:
            for e in criterion_evidence:
                evidence_lines.append(
                    f"- {e.get('file_path', 'N/A')} ({e.get('line_range', 'N/A')}): {e.get('observation', '')}"
                )
        else:
            evidence_lines.append("- No cited evidence.")
        evidence_lines.append("")

    return {
        "score": score,
        "feedback": "\n".join(feedback_lines),
        "evidence": "\n".join(evidence_lines).strip(),
    }
```

This drops `build_practice_graph`, `give_practice_feedback`, and `submit_attempt` entirely (all three existed only to route between the practice and official pipelines by attempt count, via the now-deleted `attempt_tracker.py`).

- [x] **Step 10: Run the new and existing agent tests to verify they pass**

Run: `poetry run pytest tests/test_grade_formatting.py tests/test_gather_node.py -v`
Expected: PASS (3 tests total).

- [x] **Step 11: Fix `src/visualize_graphs.py`, which imports the now-deleted `build_practice_graph`**

In `src/visualize_graphs.py`, replace the `main()` function:

```python
def main():
    from src.agent.graph import build_grading_graph
    from src.chatbot.graph import build_chat_graph

    save_graph_image(build_grading_graph(), "grading_graph.png")
    save_graph_image(build_chat_graph(), "chatbot_graph.png")
```

(This drops the `practice_graph.png` line along with the deleted `build_practice_graph`.)

- [x] **Step 12: Run the full test suite to verify nothing else broke**

Run: `poetry run pytest -v`
Expected: PASS (all tests, including `tests/test_diagram_reader.py`, unaffected by this task).

- [x] **Step 13: Commit**

```bash
git add src/agent/state.py src/agent/nodes.py src/agent/graph.py src/visualize_graphs.py tests/test_gather_node.py tests/test_grade_formatting.py
git commit -m "$(cat <<'EOF'
Trim grading graph to the official gather->format->verify pipeline only

Drops the attempt-count-routed practice-feedback pipeline (dead once
every trial goes through the same AI-drafted-then-judge-reviewed cycle
per the platform spec) and adds format_grade_response(), which POST
/grade will use to produce its exact {score, feedback, evidence} shape.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Make the chatbot stateless — accept `history` instead of `thread_id`/`InMemorySaver`

**Files:**
- Modify: `src/chatbot/graph.py`
- Modify: `src/chatbot/chatbot.py`
- Test: `tests/test_chatbot_graph.py` (new)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `src.chatbot.graph.ask(question: str, history: list[dict] | None = None) -> str` — `history` items are `{"role": "user" | "assistant", "content": str}`, oldest first. No `thread_id` parameter.
  - `src.chatbot.chatbot.answer_question(question: str, history: list[dict] | None = None) -> str` — thin wrapper around `ask()`, same signature.

The chatbot's short-term memory used to come from `checkpointer=InMemorySaver()` keyed by `thread_id` — per the spec, the assistant must keep no conversation state at all; victoris-backend persists every message and passes the relevant recent history back on each call. `chat_node`'s actual logic (retrieve → grade relevance → answer using the full message list) is unchanged; only how the message list gets built changes.

- [x] **Step 1: Write the failing test**

Create `tests/test_chatbot_graph.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot import graph


class _FakeDoc:
    def __init__(self, section, content):
        self.metadata = {"section": section}
        self.page_content = content


class _FakeRetriever:
    def invoke(self, _question):
        return [_FakeDoc("Team Size", "Teams must have 3 to 5 members.")]


class _FakeRelevanceLlm:
    def invoke(self, _prompt):
        class _Response:
            content = "[0]"
        return _Response()


class _FakeChatLlm:
    def __init__(self):
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content="Teams must have 3 to 5 members.")


def test_ask_sends_supplied_history_to_the_llm(monkeypatch):
    fake_chat_llm = _FakeChatLlm()
    llms = iter([_FakeRelevanceLlm(), fake_chat_llm])

    monkeypatch.setattr(graph, "_get_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(graph, "get_llm", lambda **_kwargs: next(llms))

    history = [
        {"role": "user", "content": "What is CodeRefine?"},
        {"role": "assistant", "content": "A system-design track."},
    ]

    answer = graph.ask("How many members per team?", history=history)

    assert answer == "Teams must have 3 to 5 members."
    sent_human_messages = [m for m in fake_chat_llm.received_messages if isinstance(m, HumanMessage)]
    assert [m.content for m in sent_human_messages] == [
        "What is CodeRefine?", "How many members per team?",
    ]
```

- [x] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/test_chatbot_graph.py -v`
Expected: FAIL — `ask()` currently requires/uses a `thread_id` and a checkpointer, so this call either errors or the assertion on `sent_human_messages` fails (the current `ask()` never mixes in caller-supplied history at all).

- [x] **Step 3: Rewrite `src/chatbot/graph.py`**

Replace the full contents of `src/chatbot/graph.py` with:

```python
"""
The team chatbot, as a small LangGraph graph. Stateless by design: the
caller (victoris-backend) supplies whatever conversation history it wants
remembered on each call -- see
docs/specs/2026-09-22-coderefine-judging-platform.md, POST /chat.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import re

import json

from src.chatbot.state import ChatState
from src.chatbot.prompts import CHATBOT_SYSTEM_PROMPT
from src.ingestion.retriever import load_retriever
from src.agent.llm import get_llm

_retriever = None

MAX_CHARS_PER_CHUNK = 800

_PROCEDURAL_KEYWORDS = [
    "regist", "sign up", "sign-up", "signup", "deadline", "apply",
    "application", "submit", "submission process", "contact form",
    "portal", "fee", "enroll", "how to join",
    # Arabic equivalents, since this chatbot has been tested bilingually
    "تسجيل", "التسجيل", "التقديم", "الاشتراك", "موعد نهائي",
]

_GENERAL_SECTION_PATTERN = re.compile(
    r"\*\*General (?:guidance|explanation)[^\n]*\*\*.*?(?=\n\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_speculative_procedure_sections(answer: str) -> str:
    def maybe_strip(match):
        block = match.group(0)
        if any(keyword.lower() in block.lower() for keyword in _PROCEDURAL_KEYWORDS):
            return ""  # drop the whole speculative block
        return block  # genuine concept explanations pass through untouched

    return _GENERAL_SECTION_PATTERN.sub(maybe_strip, answer).strip()


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = load_retriever()
    return _retriever


def _grade_relevance(question: str, docs: list) -> list:
    """
    Retrieval finds chunks that are SIMILAR in meaning to the question
    that's not the same as chunks that actually ANSWER it. A compound or
    oddly-phrased question can retrieve chunks that are topically related
    but don't directly address what was asked. This asks the LLM to check
    each retrieved chunk against the actual question before any of them
    get used to answer filtering out anything that doesn't genuinely
    help, instead of assuming "retrieved" means "relevant."
    """
    if not docs:
        return []

    numbered = "\n\n".join(
        f"[{i}] Section: {d.metadata['section']}\n{d.page_content[:MAX_CHARS_PER_CHUNK]}"
        for i, d in enumerate(docs)
    )
    prompt = f"""Question: {question}

Below are numbered sections retrieved for this question. For EACH one,
decide: could this section plausibly help answer the question, even if
the wording is different from how the question was phrased? Be
INCLUSIVE, not strict a paraphrase, a synonym, or a section that
covers part of the answer should be marked relevant. Only exclude a
section if it is genuinely about something else entirely.

{numbered}

Return ONLY a JSON list of the indices (integers) of sections that could
plausibly help. Return an empty list [] only if NONE of them are
remotely related to the question. Example: [0, 2]"""

    llm = get_llm(temperature=0.0)  # deterministic for a grading/filtering task
    response = llm.invoke(prompt)
    try:
        relevant_indices = json.loads(response.content)
        return [docs[i] for i in relevant_indices if isinstance(i, int) and 0 <= i < len(docs)]
    except (json.JSONDecodeError, TypeError):
        # If grading itself fails to parse, fail toward including everything
        # retrieved rather than silently answering from nothing.
        return docs


def chat_node(state: ChatState) -> dict:
    """
    The only node in this graph. Runs once per call: retrieves relevant
    rules for the LATEST message, but sends the FULL message list (caller-
    supplied history + the new question) to the LLM that combination is
    what lets it answer follow-ups correctly while still grounding answers
    in retrieved rules.
    """
    latest_question = state["messages"][-1].content

    retriever = _get_retriever()
    docs = retriever.invoke(latest_question)
    relevant_docs = _grade_relevance(latest_question, docs)

    if not relevant_docs and docs:
        relevant_docs = docs[:1]

    if relevant_docs:
        context = "\n\n".join(
            f"[{d.metadata['section']}]\n{d.page_content[:MAX_CHARS_PER_CHUNK]}" for d in relevant_docs
        )
    else:
        context = ""

    system_message = SystemMessage(content=CHATBOT_SYSTEM_PROMPT.format(context=context))
    llm = get_llm(temperature=0.2)

    response = llm.invoke([system_message] + state["messages"])
    response.content = _strip_speculative_procedure_sections(response.content)
    return {"messages": [response]}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    # No checkpointer: the caller supplies whatever history it wants
    # remembered on every call, instead of this service remembering it.
    return graph.compile()


_chat_app = None


def _get_chat_app():
    global _chat_app
    if _chat_app is None:
        _chat_app = build_chat_graph()
    return _chat_app


def ask(question: str, history: list[dict] | None = None) -> str:
    """
    history: prior turns as [{"role": "user" | "assistant", "content": str}, ...],
    oldest first -- exactly what POST /chat's request body carries. No
    thread_id: this call is stateless, all context comes from history.
    """
    messages = []
    for turn in history or []:
        message_cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(message_cls(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    app = _get_chat_app()
    result = app.invoke({"messages": messages})
    return result["messages"][-1].content
```

This drops the `InMemorySaver` import/checkpoint and the `log_question` import/call (the question/answer audit log is deleted in Task 4 — victoris-backend now persists every chat message itself).

- [x] **Step 4: Rewrite `src/chatbot/chatbot.py`**

Replace the full contents of `src/chatbot/chatbot.py` with:

```python
"""
Thin backward-compatible entry point. The real logic lives in graph.py.
"""

from src.chatbot.graph import ask


def answer_question(question: str, history: list[dict] | None = None) -> str:
    return ask(question, history=history)


if __name__ == "__main__":
    # Loading .env here specifically, because running this file directly
    # (poetry run python -m src.chatbot.chatbot) skips main.py entirely --
    # and main.py is normally what loads the .env file. Any file with its
    # own standalone test block needs to handle this itself.
    from dotenv import load_dotenv
    load_dotenv()

    # Quick manual test -- two related questions, second one is a follow-up
    # that only makes sense when the caller passes the first turn back in
    # as history (this service no longer remembers it on its own).
    first_question = "How many members can be on a team?"
    first_answer = answer_question(first_question)
    print(f"Q: {first_question}")
    print(f"A: {first_answer}\n")

    second_question = "And what age do they need to be?"
    history = [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": first_answer},
    ]
    print(f"Q: {second_question}")
    print(f"A: {answer_question(second_question, history=history)}")
```

- [x] **Step 5: Run the test to verify it passes**

Run: `poetry run pytest tests/test_chatbot_graph.py -v`
Expected: PASS.

- [x] **Step 6: Run the full test suite to verify nothing else broke**

Run: `poetry run pytest -v`
Expected: PASS (all tests from Task 1 plus this one).

- [x] **Step 7: Commit**

```bash
git add src/chatbot/graph.py src/chatbot/chatbot.py tests/test_chatbot_graph.py
git commit -m "$(cat <<'EOF'
Make the chatbot stateless: accept history instead of thread_id

Drops InMemorySaver/thread_id checkpointing and the questions.jsonl
audit log call. The caller now supplies whatever history it wants
remembered on every call, matching POST /chat's contract -- this
service itself keeps no conversation state.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rewrite `src/api/main.py` as the stateless, bearer-token-authenticated API

**Files:**
- Create: `src/api/main.py` (full rewrite)
- Delete: `src/api/auth.py`
- Delete: `src/api/job_queue.py`
- Test: `tests/test_api_main.py` (new)

**Interfaces:**
- Consumes:
  - `src.agent.graph.grade_repo`, `src.agent.graph.format_grade_response` (Task 1)
  - `src.chatbot.chatbot.answer_question` (Task 2)
- Produces:
  - `src.api.main.app` — the FastAPI instance, routes `POST /chat`, `POST /grade`, `GET /health`.
  - `src.api.main.require_service_token` — a FastAPI dependency; raises 401 when the `Authorization: Bearer <token>` header is missing or doesn't match the `SERVICE_AUTH_TOKEN` env var.
  - Env var `SERVICE_AUTH_TOKEN` (new; replaces `TEAM_PASSWORD`/`JUDGE_PASSWORD`, updated in `.env.example` in Task 5).

`job_queue.py` (the single-worker background thread that paced submissions) and `auth.py` (password login + in-memory session tokens) both existed only to serve the old `/login/*` and `/submit` endpoints. Neither has a role in the new synchronous, bearer-token-only API, so both are deleted here rather than kept as unused code. CORS middleware and `slowapi` rate limiting are also dropped: the spec is explicit that "this service is never called directly by a browser", so there's no browser origin to allow, and rate limiting an internal service reachable only by victoris-backend behind a shared secret isn't something the spec asks for.

- [x] **Step 1: Write the failing tests**

Create `tests/test_api_main.py`:

```python
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


@pytest.fixture(autouse=True)
def _service_token(monkeypatch):
    monkeypatch.setenv("SERVICE_AUTH_TOKEN", "test-token")


def _client():
    return TestClient(api_main.app)


def test_health_requires_no_token():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_missing_token():
    response = _client().post("/chat", json={"question": "hi"})
    assert response.status_code == 401


def test_chat_rejects_wrong_token():
    response = _client().post(
        "/chat", json={"question": "hi"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_chat_returns_answer(monkeypatch):
    monkeypatch.setattr(api_main, "answer_question", lambda question, history=None: f"echo: {question}")

    response = _client().post(
        "/chat",
        json={"question": "How many members per team?", "history": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "echo: How many members per team?"}


def test_grade_returns_score_feedback_evidence(monkeypatch):
    monkeypatch.setattr(api_main, "grade_repo", lambda repo_url: {
        "final_scorecard": [{
            "criterion": "X", "score_percent": 10, "justification": "j",
            "confidence": "high", "evidence": [],
        }],
        "verification_notes": "All evidence checked out.",
        "repo_error": None,
    })

    response = _client().post(
        "/grade",
        json={"repoUrl": "https://github.com/owner/repo"},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 10
    assert "X" in body["feedback"]
    assert "X" in body["evidence"]


def test_grade_rejects_missing_token():
    response = _client().post("/grade", json={"repoUrl": "https://github.com/owner/repo"})
    assert response.status_code == 401


def test_grade_returns_422_on_repo_error(monkeypatch):
    monkeypatch.setattr(api_main, "grade_repo", lambda repo_url: {
        "final_scorecard": [],
        "verification_notes": "",
        "repo_error": "Could not access repo: 404",
    })

    response = _client().post(
        "/grade",
        json={"repoUrl": "https://github.com/owner/missing"},
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": "Could not access repo: 404"}
```

- [x] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/test_api_main.py -v`
Expected: FAIL — `src.api.main` still imports `src.api.auth`/`src.api.job_queue`, has no `/grade` route, and `/chat`/`/grade` aren't token-gated yet.

- [x] **Step 3: Delete `src/api/auth.py` and `src/api/job_queue.py`**

```bash
git rm src/api/auth.py src/api/job_queue.py
```

- [x] **Step 4: Write `src/api/main.py`**

Replace the full contents of `src/api/main.py` with:

```python
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
```

Note the top-level imports of `answer_question`, `grade_repo`, and `format_grade_response` (rather than importing inside each route function, which the old `main.py` did) -- this matches the rest of the repo's test convention of monkeypatching a name directly on the consuming module (e.g. `monkeypatch.setattr(nodes, "find_submission_files", ...)` in `tests/test_gather_node.py`), which only works if the name is bound at module import time.

- [x] **Step 5: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_api_main.py -v`
Expected: PASS (7 tests).

- [x] **Step 6: Run the full test suite to verify nothing else broke**

Run: `poetry run pytest -v`
Expected: PASS (every test from Tasks 1-3).

- [x] **Step 7: Commit**

```bash
git add src/api/main.py tests/test_api_main.py
git commit -m "$(cat <<'EOF'
Rewrite the API as a stateless, bearer-token-authenticated service

Deletes src/api/auth.py (password/session login) and
src/api/job_queue.py (background submission worker) -- neither has a
role once /grade is synchronous and auth is a single static bearer
token. New src/api/main.py exposes exactly POST /chat, POST /grade,
GET /health, matching the spec's API contracts section field-for-field.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Delete the remaining dead persistence/CLI/website code

**Files:**
- Delete: `src/agent/attempt_tracker.py`
- Delete: `src/agent/practice_store.py`
- Delete: `src/agent/review_queue.py`
- Delete: `src/agent/report.py`
- Delete: `src/logging_utils.py`
- Delete: `src/batch_grade.py`
- Delete: `logs/review_queue.json`
- Delete: `web/` (entire directory)
- Modify: `src/main.py`

**Interfaces:**
- Consumes: `src.chatbot.chatbot.answer_question(question, history=None)` (Task 2) -- the trimmed CLI's only remaining dependency.
- Produces: no new importable interface; this task's deliverable is "the repo tree matches the trimmed scope, and nothing still imports a deleted module."

By the end of Task 3, nothing in `src/agent/`, `src/api/`, or `src/chatbot/` references any of these modules any more. The only remaining consumer is the CLI (`src/main.py`), whose `submit`/`review`/`approve`/`release`/`report` subcommands only ever existed to drive the now-deleted attempt-tracking and review-queue workflow. `logs/review_queue.json` is a stray, git-tracked leftover from an earlier JSON-file version of the review queue (predates the SQLite rewrite; not covered by `.gitignore`'s `logs/*.jsonl`/`logs/review_queue.db` rules since its extension is plain `.json`) -- dead data with no reader left in the trimmed service. `web/index.html` is entirely the old team/judge login UI calling `/login/team`, `/login/judge`, `/submit`, `/queue`, `/approve`, `/release` (confirmed by reading it) -- none of those routes exist any more, and `src/api/main.py` no longer serves it at `/`.

- [x] **Step 1: Delete the persistence/audit/batch modules and the stray JSON file**

```bash
git rm src/agent/attempt_tracker.py src/agent/practice_store.py src/agent/review_queue.py src/agent/report.py src/logging_utils.py src/batch_grade.py logs/review_queue.json
```

- [x] **Step 2: Delete the old website**

```bash
git rm -r web/
```

- [x] **Step 3: Trim `src/main.py` to just the `chat` subcommand**

Replace the full contents of `src/main.py` with:

```python
"""
The single entry point for running this project locally.

Usage:
    poetry run python -m src.main chat
"""

import argparse

from dotenv import load_dotenv

load_dotenv()  # reads .env so GROQ_API_KEY / GITHUB_TOKEN are available


def run_chat() -> None:
    """
    Local manual testing only -- the real chat surface is POST /chat
    (src/api/main.py). This REPL keeps its own history list and passes it
    on every call, exactly like victoris-backend will, since the chatbot
    itself remembers nothing between calls.
    """
    from src.chatbot.chatbot import answer_question

    history: list[dict] = []
    print("Team support chatbot. Type 'exit' to quit.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        answer = answer_question(question, history=history)
        print(f"A: {answer}\n")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeRefine AI assistant -- local dev CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat", help="Start the team support chatbot")

    args = parser.parse_args()
    if args.command == "chat":
        run_chat()


if __name__ == "__main__":
    main()
```

This CLI was never covered by an automated test (no existing test imports `src.main`), so no test file changes here -- the `submit`/`review`/`approve`/`release`/`report` subcommands are simply gone along with the modules they drove.

- [x] **Step 4: Grep for any remaining references to the deleted modules**

Run: `grep -rn "attempt_tracker\|practice_store\|review_queue\|agent\.report\|logging_utils\|batch_grade" src/ tests/`
Expected: no output (empty). If anything matches, it's a leftover import or comment that must be fixed before continuing -- fix it and re-run.

- [x] **Step 5: Run the full test suite to verify nothing broke**

Run: `poetry run pytest -v`
Expected: PASS (every test from Tasks 1-3; this task touches no test-covered code paths).

- [x] **Step 6: Commit**

```bash
git add -A src/main.py
git commit -m "$(cat <<'EOF'
Delete the review-queue/attempt-tracking/website code and trim the CLI

Removes attempt_tracker.py, practice_store.py, review_queue.py,
report.py, logging_utils.py, batch_grade.py, a stray leftover
review_queue.json, and the old team/judge login website -- all of
victoris-backend's job now, per the platform spec. src/main.py keeps
only its `chat` subcommand for local manual testing.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update dependencies, `.env.example`, and `docker-compose.yml` for the trimmed scope

**Files:**
- Modify: `pyproject.toml`
- Modify: `poetry.lock` (regenerated, not hand-edited)
- Modify: `.env.example`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: nothing new (verifies the dependency/config surface matches Tasks 1-4's finished code).
- Produces: nothing importable; this task's deliverable is "the project's declared dependencies and runtime config match what the trimmed code actually uses."

`slowapi` (rate limiting) and `python-multipart` (form/file upload parsing) are now unused: `slowapi` was only imported by the old `src/api/main.py` (rewritten in Task 3, dropped it), and a grep across `src/` finds no `UploadFile`/`Form(...)`/`python_multipart` usage anywhere, present or past -- `python-multipart` was declared but never actually used. `TEAM_PASSWORD`/`JUDGE_PASSWORD` are replaced by `SERVICE_AUTH_TOKEN` (Task 3). `docker-compose.yml`'s `./logs:/app/logs` volume existed to persist the SQLite DBs and JSONL logs deleted in Task 4 -- with no runtime code writing into `logs/` any more (only the manually-run `src/visualize_graphs.py` dev tool does, and its output doesn't need to survive a container restart), that volume mount is removed. The `chroma_db` and `hf_cache` volumes stay untouched, per the spec.

- [ ] **Step 1: Grep to confirm `slowapi` and `python-multipart` are unused**

Run: `grep -rn "slowapi\|multipart\|UploadFile" src/`
Expected: no output (empty) -- Task 3 already removed the only `slowapi` import, and `python-multipart` was never imported anywhere.

- [ ] **Step 2: Remove the two unused dependencies from `pyproject.toml`**

In `src/../pyproject.toml`'s `[tool.poetry.dependencies]` block, delete these two lines:

```toml
slowapi = "^0.1.10"
```

and

```toml
python-multipart = "^0.0.12"
```

Leave every other dependency (`fastapi`, `uvicorn`, `pydantic`, `langgraph`, `langchain`, `langchain-groq`, `langchain-chroma`, `langchain-huggingface`, `sentence-transformers`, `chromadb`, `pygithub`, `python-dotenv`, `pymupdf`) and the `[tool.poetry.group.dev.dependencies]` block (`pytest`, `black`) unchanged.

- [ ] **Step 3: Regenerate the lock file and reinstall**

Run: `poetry lock`
Run: `poetry install`
Expected: both succeed; `poetry.lock` no longer lists `slowapi` or `python-multipart`.

- [ ] **Step 4: Update `.env.example`**

Replace the full contents of `.env.example` with:

```
GROQ_API_KEY=
GITHUB_TOKEN=
SERVICE_AUTH_TOKEN=
```

- [ ] **Step 5: Update `docker-compose.yml`**

Replace the full contents of `docker-compose.yml` with:

```yaml
services:
  app:
    build: .
    ports:
      - "5003:7860"
    env_file: .env
    environment:
      HF_HOME: /hf
    volumes:
      # Chroma vector store (knowledge base) + BGE-M3 model cache survive
      # restarts. No SQLite/JSONL volume: this service is stateless --
      # victoris-backend owns all trial/chat persistence (see
      # docs/specs/2026-09-22-coderefine-judging-platform.md).
      - chroma_db:/app/data/chroma_db
      - hf_cache:/hf
    # build the knowledge base only if the volume is empty (first run)
    command: >
      sh -c "[ -n \"$(ls -A data/chroma_db 2>/dev/null)\" ] ||
      poetry run python -m src.ingestion.build_knowledge_base;
      poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 7860"
    restart: unless-stopped

volumes:
  chroma_db:
  hf_cache:
```

This drops the `ALLOWED_ORIGIN` environment variable (CORS middleware is gone -- Task 3) and the `./logs:/app/logs` volume line, and updates the volumes comment.

- [ ] **Step 6: Run the full test suite to verify nothing broke**

Run: `poetry run pytest -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml poetry.lock .env.example docker-compose.yml
git commit -m "$(cat <<'EOF'
Drop unused deps and stale config for the trimmed service

Removes slowapi and python-multipart (both unused once the old
password-login/rate-limited API and file-upload plumbing are gone),
replaces TEAM_PASSWORD/JUDGE_PASSWORD with SERVICE_AUTH_TOKEN, and
drops docker-compose.yml's logs/ volume and ALLOWED_ORIGIN now that
this service is stateless and never called from a browser. The Chroma
vector store volume is untouched.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update documentation for the trimmed scope

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `SETUP_GUIDE.md`
- Delete: `project_explain.md`
- Modify: `src/README.md`
- Modify: `src/api/README.md`
- Modify: `src/agent/README.md`
- Modify: `src/chatbot/README.md`

**Interfaces:**
- Consumes: the finished state of Tasks 1-5 (this task only describes it).
- Produces: nothing importable; documentation only.

`project_explain.md` is a 156-line narrative walkthrough of the team/judge/practice-trial/website flow -- that whole flow now belongs to victoris-backend and coderefine-platform, not this repo, so rewriting it here would just be describing another repo's responsibility. It's deleted rather than rewritten; `CLAUDE.md` and `README.md` are now short enough to cover the trimmed service on their own. `src/ingestion/README.md` and `src/tools/README.md` describe modules kept unchanged and stay as they are.

- [ ] **Step 1: Delete `project_explain.md`**

```bash
git rm project_explain.md
```

- [ ] **Step 2: Rewrite `CLAUDE.md`**

Replace the full contents of `CLAUDE.md` with:

```markdown
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
```

- [ ] **Step 3: Rewrite `README.md`**

Replace the full contents of `README.md` with:

```markdown
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
```

- [ ] **Step 4: Rewrite `SETUP_GUIDE.md`**

Replace the full contents of `SETUP_GUIDE.md` with:

```markdown
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
```

- [ ] **Step 5: Rewrite `src/README.md`**

Replace the full contents of `src/README.md` with:

```markdown
# src/, top level files

`main.py`: the local dev CLI. `chat` is the only subcommand -- an interactive REPL over the same stateless `answer_question(question, history)` that `POST /chat` calls, keeping its own `history` list and passing it back on every turn.

`visualize_graphs.py`: generates PNG image files of the LangGraph graphs in this project (the grading graph, the chatbot graph). Uses LangGraph's built-in `draw_mermaid_png()`, which needs internet access but no local install. Saves images into `logs/`.

`debug_retrieval.py`: a small diagnostic tool. Shows exactly which knowledge-base chunks get retrieved for a given question, with no LLM involved. Useful when the chatbot seems to be missing something that should be in the rules document, since this tells you whether it's a retrieval problem or something else.
```

- [ ] **Step 6: Rewrite `src/api/README.md`**

Replace the full contents of `src/api/README.md` with:

```markdown
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
```

- [ ] **Step 7: Rewrite `src/agent/README.md`**

Replace the full contents of `src/agent/README.md` with:

```markdown
# src/agent/

The grading agent. Reads a submitted repo and produces a scored, evidence-cited draft -- always through the same three-step pipeline, for every trial.

## Files

`rubric.py`: the five real judging criteria and their weights (15/20/20/25/20). No functions, just data. Bonus is scored separately by a judge in victoris-dashboard, never by this agent.

`llm.py`: creates the Groq clients. `get_llm()` is the text client for grading and chat. `get_vision_llm()` is a separate Qwen vision client used only to make factual observations about committed image/PDF diagrams. Both use `GROQ_API_KEY`.

`state.py`: `GradingState`, the shape of data that flows through the pipeline -- what's known at the start (`repo_url`), what `gather` fills in (`raw_notes`, and `repo_error` if the repo couldn't be reached at all), what `format` fills in (`draft_scorecard`), what `verify` fills in (`final_scorecard`, `verification_notes`). No logic in this file, just the data shape.

`nodes.py`: the actual reasoning logic, as three functions:
- `gather_node`, reads the repo's README, Deep Dives file, BOTE file, and architecture diagram, and produces raw observations (no scoring). Committed PNG/JPG/WebP diagrams and the first five PDF pages add factual vision evidence; the Excalidraw parser handles `.excalidraw` files. Includes prompt-injection defenses, since this content is written by the team being graded and shouldn't be trusted as instructions. Sets `repo_error` in its returned dict if the repo itself couldn't be reached.
- `format_node`, turns those observations into a scored draft, citing evidence by ID rather than copied text so verification doesn't break on paraphrasing.
- `verify_node`, a plain code check (no LLM call) that confirms cited evidence actually exists, and flags any justification that looks like it invented a scoring threshold not in the rubric.

`graph.py`: wires the three functions above into one runnable LangGraph flow. `grade_repo(repo_url)` is the one function `POST /grade` calls -- runs the pipeline, returns the full final state. `format_grade_response(final_scorecard, verification_notes)` is a separate pure function that reshapes that internal state into `POST /grade`'s exact `{score, feedback, evidence}` response.

## Nothing in this folder runs on its own

Everything here gets called from `src/api/main.py` (or `src/visualize_graphs.py` for diagram generation). There's no reason to run any of these files directly, except `state.py`, `rubric.py`, and `llm.py`, which aren't runnable at all since they define things other files use.
```

- [ ] **Step 8: Fix `src/chatbot/README.md`**

Replace the full contents of `src/chatbot/README.md` with:

```markdown
# src/chatbot/

The team support chatbot. Answers questions using only the rules document (via `src/ingestion/`) no connection to the grading agent at all. Stateless: every call takes the full conversation history it needs as a parameter, and remembers nothing on its own.

## Files

**`prompts.py`** the system prompt, and it's doing more work than it looks like. Draws a hard line between three things: facts stated in the actual rules (must be quoted accurately, never embellished), general concepts explained from common knowledge (allowed, but must be clearly labeled as not an official rule), and this organization's real procedures like registration or deadlines (never guessed at, even when labeled if it's not in the rules doc, the answer is "not specified, contact the organizers"). Includes worked examples of right vs. wrong answers, since plain instructions alone weren't reliably followed in testing.

**`state.py`** defines the per-call message shape. One field, `messages`, using LangGraph's `add_messages` so new messages get appended within a single graph invocation instead of replacing it.

**`graph.py`**: the actual chatbot logic, in order: retrieve candidate chunks for the latest question, grade each one for genuine relevance to the question (a separate LLM call, since a chunk being retrieved because it's topically similar isn't the same as it actually answering what was asked), then answer using only chunks that passed grading. If grading rejects everything, the top raw retrieval result is used as a fallback rather than leaving the answer with nothing. `ask(question, history)` builds the message list from the caller-supplied `history` (oldest first) plus the new question, and sends the whole thing to the LLM alongside the retrieved context. Also caps how much of any single retrieved chunk gets used, so one oversized chunk can't blow through the LLM's rate limit by itself.

**`chatbot.py`** a thin wrapper exposing `answer_question(question, history=None)`, which is what everything else (the CLI, the API) actually calls. Also runnable directly for a quick manual test.

## No memory of its own

`ask()`/`answer_question()` take `history` as a plain argument -- a list of `{"role": "user" | "assistant", "content": str}` dicts, oldest first. There is no `thread_id` and nothing is persisted here: the caller (victoris-backend) is responsible for storing every message and passing back whatever it wants remembered on the next call.
```

- [ ] **Step 9: Run the full test suite one more time to confirm the doc-only changes broke nothing**

Run: `poetry run pytest -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md README.md SETUP_GUIDE.md src/README.md src/api/README.md src/agent/README.md src/chatbot/README.md
git rm project_explain.md
git commit -m "$(cat <<'EOF'
Update documentation for the trimmed, stateless API surface

CLAUDE.md, README.md, SETUP_GUIDE.md, and the per-folder READMEs now
describe the actual POST /chat, POST /grade, GET /health surface and
bearer-token auth instead of the removed login/practice-trial/judge-
review flow. project_explain.md's narrative walkthrough of that flow
now belongs to victoris-backend/coderefine-platform, not this repo, so
it's deleted rather than rewritten.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification sweep

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Run the full test suite**

Run: `poetry run pytest -v`
Expected: PASS, all tests across `tests/test_diagram_reader.py`, `tests/test_gather_node.py`, `tests/test_grade_formatting.py`, `tests/test_chatbot_graph.py`, `tests/test_api_main.py`.

- [ ] **Step 2: Confirm the app actually imports and boots**

Run: `poetry run python -c "from src.api.main import app; print([r.path for r in app.routes])"`
Expected: prints a list of paths including `/chat`, `/grade`, `/health` (plus FastAPI's built-in `/openapi.json`, `/docs`, `/redoc`), with no import errors.

- [ ] **Step 3: Grep for any stray references to removed endpoints or concepts**

Run: `grep -rn "login/team\|login/judge\|thread_id\|InMemorySaver\|TEAM_PASSWORD\|JUDGE_PASSWORD\|session_token" src/ tests/ *.md docker-compose.yml Dockerfile .env.example`
Expected: no output. (`thread_id`/`InMemorySaver` should not appear anywhere any more; the rest were the old auth's vocabulary.)

- [ ] **Step 4: Confirm no code still imports a deleted module**

Run: `grep -rn "from src.api.auth\|from src.api.job_queue\|from src.agent.attempt_tracker\|from src.agent.practice_store\|from src.agent.review_queue\|from src.agent.report\|from src.logging_utils\|from src.batch_grade\|import src.batch_grade" src/ tests/`
Expected: no output.

- [ ] **Step 5: Confirm the repo tree matches the "keep" list from the spec**

Run: `git status` (should be clean -- everything from Tasks 1-6 already committed) and `git ls-files src/ | sort`
Expected: `src/api/main.py`, `src/agent/{graph,nodes,state,rubric,llm}.py`, `src/chatbot/{graph,chatbot,state,prompts}.py`, `src/ingestion/*`, `src/tools/*`, `src/main.py`, `src/visualize_graphs.py`, `src/debug_retrieval.py` -- and none of `src/api/auth.py`, `src/api/job_queue.py`, `src/agent/attempt_tracker.py`, `src/agent/practice_store.py`, `src/agent/review_queue.py`, `src/agent/report.py`, `src/logging_utils.py`, `src/batch_grade.py`.

No commit for this task -- it only confirms the prior six tasks' commits already leave the repo correct. If any check above fails, fix it in a follow-up commit on the same branch before considering this plan done.

---

## Self-Review

**1. Spec coverage** (against the "AI assistant (CodeRefine-Judging-Assistant) — trimmed scope" section and its API contracts):
- "Keep: `src/chatbot/`, `src/ingestion/`, `src/agent/` grading graph, `src/tools/`, `src/agent/llm.py`" → untouched except the two required interface changes (Tasks 1, 2); `src/ingestion/`, `src/tools/`, `src/agent/llm.py` not modified anywhere in this plan.
- "Remove entirely: `src/api/auth.py`, `attempt_tracker.py`, `review_queue.py`, `practice_store.py`, `batch_grade.py`, all SQLite persistence, `web/`, `logs/questions.jsonl`/`logs/scorecards.jsonl` audit logging" → Tasks 3 and 4 (auth.py/job_queue.py in Task 3 since main.py's rewrite is what orphans them; the rest in Task 4). The JSONL logs never separately existed as files to delete (append-only, gitignored) -- their writer, `logging_utils.py`, is deleted in Task 4, and the `log_question`/`log_scorecard`/`log_practice_feedback` call sites are removed in Tasks 1, 2, and 4 along with their callers.
- "`POST /chat { question, history? } → { answer }`, no thread_id/InMemorySaver" → Task 2 (graph.py rewrite) + Task 3 (route).
- "`POST /grade { repoUrl } → { score, feedback, evidence }`, 422 on repo unreachable" → Task 1 (`format_grade_response`, `repo_error`) + Task 3 (route, 422 handling with the exact `{error}` body shape).
- "`GET /health` stays" → Task 3.
- "Auth: single static bearer token, every route except /health" → Task 3 (`require_service_token`).
- "`ponytail:` synchronous, no job queue, for now" → carried into the plan's Global Constraints and as a matching code comment in Task 3's `/grade` handler.
- Field-name exactness (`repoUrl`, `question`, `history`, `role`/`content`, `answer`, `score`/`feedback`/`evidence`, `error`, `status`) → verified against every Pydantic model and response dict in Task 3; the 422 body was specifically fixed to return `{"error": ...}` rather than FastAPI's default `{"detail": ...}`.

**2. Placeholder scan:** every step above contains full, runnable code or an exact shell command with expected output -- no "TBD", no "add appropriate handling", no "similar to Task N" cross-references without the actual code repeated in place.

**3. Type consistency:** `GradingState` (Task 1) has no `team_name` and gains `repo_error: Optional[str]`; `grade_repo(repo_url: str) -> dict` and `format_grade_response(final_scorecard: list[dict], verification_notes: str) -> dict` (Task 1) are the exact names/signatures imported by `src/api/main.py` in Task 3. `ask(question: str, history: list[dict] | None = None) -> str` and `answer_question(question: str, history: list[dict] | None = None) -> str` (Task 2) are the exact names/signatures imported by `src/api/main.py` (Task 3) and `src/main.py` (Task 4). No task references a function or field name not defined in an earlier task.
