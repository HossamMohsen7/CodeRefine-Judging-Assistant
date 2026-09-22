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


def test_grade_returns_score_scorecard_verification_notes(monkeypatch):
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
    assert body["verificationNotes"] == "All evidence checked out."
    assert body["scorecard"] == [
        {"criterion": "X", "scorePercent": 10, "confidence": "high", "justification": "j"}
    ]


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
