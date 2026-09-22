import json

from src.agent import nodes


class _Response:
    content = json.dumps({"observations": [{
        "file_path": "README.md",
        "line_range": "1-1",
        "excerpt": "A service is described.",
        "observation": "The README describes a service.",
    }]})


class _TextLlm:
    def invoke(self, _prompt):
        return _Response()


def test_gather_appends_committed_diagram_evidence(monkeypatch):
    monkeypatch.setattr(nodes, "find_submission_files", lambda _repo: {
        "readme": "README.md", "deep_dives": None, "bote": None,
        "excalidraw": None, "images": ["diagram.png"], "pdfs": [],
    })
    monkeypatch.setattr(nodes, "fetch_readme", lambda _repo: "README")
    monkeypatch.setattr(nodes, "fetch_file_bytes", lambda _repo, _path: b"image")
    monkeypatch.setattr(nodes, "find_external_resource_links", lambda _readme: [])
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: _TextLlm())
    monkeypatch.setattr(nodes, "analyze_diagram_file", lambda path, _content: [{
        "file_path": path,
        "line_range": "N/A",
        "excerpt": "API -> Database",
        "observation": "The API connects to a database.",
        "source_type": "image",
    }])

    result = nodes.gather_node({"repo_url": "https://github.com/owner/repo"})

    assert len(result["raw_notes"]) == 2
    assert result["raw_notes"][1]["source_type"] == "image"
    assert [note["id"] for note in result["raw_notes"]] == [0, 1]


def test_gather_flags_repo_error_when_repo_unreachable(monkeypatch):
    def _raise(_repo_url):
        raise RuntimeError("Could not access repo 'owner/repo': 404")

    monkeypatch.setattr(nodes, "find_submission_files", _raise)

    result = nodes.gather_node({"repo_url": "https://github.com/owner/repo"})

    assert result["repo_error"] == "Could not access repo 'owner/repo': 404"
    assert "Could not access repo" in result["raw_notes"][0]["observation"]
