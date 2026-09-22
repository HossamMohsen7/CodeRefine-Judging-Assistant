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
    into POST /grade's exact {score, scorecard, verificationNotes} response
    shape -- kept structured (not flattened to text) so the judge queue can
    render a real per-criterion table instead of parsing a paragraph.

    score: the sum of each criterion's score_percent (0-100; rubric.py's
    weights sum to 100). Bonus is judge-only and never appears here.
    scorecard: one entry per rubric criterion, judge-facing only (this
    whole response is never sent to a team -- see victoris-backend's
    coderefineDto.ts, which has a separate team-facing DTO with no AI
    fields at all). Per-criterion evidence citations are intentionally
    left out of the wire response; only the aggregate verificationNotes
    travels, matching what the judge queue actually displays.
    """
    score = sum(item.get("score_percent", 0) for item in final_scorecard)

    scorecard = [
        {
            "criterion": item["criterion"],
            "scorePercent": item.get("score_percent", 0),
            "confidence": item.get("confidence", "low"),
            "justification": item.get("justification", "No justification recorded."),
        }
        for item in final_scorecard
    ]

    return {
        "score": score,
        "scorecard": scorecard,
        "verificationNotes": verification_notes,
    }
