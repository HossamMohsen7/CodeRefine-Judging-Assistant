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
