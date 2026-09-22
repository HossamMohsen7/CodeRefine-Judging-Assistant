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
    assert result["verificationNotes"] == "All evidence checked out."
    assert result["scorecard"] == [
        {
            "criterion": "Data Model",
            "scorePercent": 15,
            "confidence": "high",
            "justification": "Covers all entities.",
        },
        {
            "criterion": "API Design",
            "scorePercent": 10,
            "confidence": "low",
            "justification": "Partial coverage.",
        },
    ]
