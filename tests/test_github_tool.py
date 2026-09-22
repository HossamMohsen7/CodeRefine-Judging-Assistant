import pytest
from src.tools.github_tool import _repo_name_from_url


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/owner/repo", "owner/repo"),
        ("https://github.com/owner/repo/", "owner/repo"),
        ("https://github.com/owner/repo.git", "owner/repo"),
        # The actual bug: a team pastes the URL straight from their browser
        # while viewing a specific branch, not the bare repo root.
        ("https://github.com/Baselhamza/seekers_/tree/main", "Baselhamza/seekers_"),
        ("https://github.com/owner/repo/tree/feature/branch-name", "owner/repo"),
        ("https://github.com/owner/repo/blob/main/README.md", "owner/repo"),
        ("https://github.com/owner/repo/pulls", "owner/repo"),
    ],
)
def test_repo_name_from_url_strips_trailing_path(url, expected):
    assert _repo_name_from_url(url) == expected


def test_repo_name_from_url_rejects_non_github_url():
    with pytest.raises(ValueError):
        _repo_name_from_url("https://gitlab.com/owner/repo")
