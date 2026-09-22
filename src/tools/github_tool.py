"""
The agent's one external tool: reading a team's submitted GitHub repo.
Kept deliberately narrow -- it only reads, it never writes back to GitHub.
"""

import os
import re
from github import Github, GithubException

# Folders that add noise, not signal, if walked -- dependency trees,
# version control internals, build output. Without excluding these, a
# large repo could burn hundreds of API calls on files that were never
# going to be relevant, and increases the odds of a false-positive
# filename match buried inside some dependency's internals.
_EXCLUDED_DIRS = {
    "node_modules", ".git", "vendor", "dist", "build",
    "__pycache__", ".venv", "venv", ".idea", ".vscode", "target",
}

# Image formats a team might commit as a diagram INSTEAD of an .excalidraw
# file. We can't parse these without a vision model (not currently wired
# in), so these are detected and flagged for manual judge review rather
# than silently never being looked at.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".webp")

# Common tools teams might use for diagrams/docs INSTEAD of Excalidraw --
# broader than just excalidraw.com, since we only had evidence for that
# one tool from a small sample of real submissions.
_EXTERNAL_TOOL_DOMAINS = [
    "excalidraw.com", "draw.io", "diagrams.net", "figma.com",
    "lucidchart.com", "miro.com", "docs.google.com", "drive.google.com",
]


def _get_client() -> Github:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set."
        )
    return Github(token)


def find_submission_files(repo_url: str) -> dict:
    """
    Locates the specific files that matter for a CodeRefine system-design
    submission. Naming varies a lot in practice -- checks the FILENAME
    only (not the full path), so files in subfolders (e.g. docs/README.md)
    are still found, not just files sitting at the repo root.

    Returns a dict with "readme", "deep_dives", "excalidraw", "bote",
    "images" (list, possibly empty), and "pdfs" (list, possibly empty).
    None for a single file not found -- the caller must treat that as
    missing evidence, not silently skip it.
    """
    file_tree = fetch_repo_file_tree(repo_url)

    def normalize(name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    def basename(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    readme_path = next(
        (f for f in file_tree if basename(f).lower().startswith("readme")), None
    )
    deep_dives_path = next(
        (f for f in file_tree if "deepdive" in normalize(basename(f))), None
    )
    excalidraw_path = next((f for f in file_tree if f.lower().endswith(".excalidraw")), None)
    bote_path = next(
        (
            f for f in file_tree
            if "backoftheenvelope" in normalize(basename(f)) or "bote" in normalize(basename(f))
        ),
        None,
    )
    # Possible diagram images and PDF design docs -- not read
    # just detected, so nothing goes silently unnoticed.
    image_paths = [f for f in file_tree if f.lower().endswith(_IMAGE_EXTENSIONS)]
    pdf_paths = [f for f in file_tree if f.lower().endswith(".pdf")]

    return {
        "readme": readme_path,
        "deep_dives": deep_dives_path,
        "excalidraw": excalidraw_path,
        "bote": bote_path,
        "images": image_paths,
        "pdfs": pdf_paths,
    }


def fetch_repo_file_tree(repo_url: str) -> list[str]:
    """
    Returns a flat list of file paths in the repo's default branch,
    skipping known-noisy directories (dependencies, build output, VCS
    internals) so large repos don't burn excessive API calls or produce
    false-positive filename matches buried inside them.
    """
    repo_name = _repo_name_from_url(repo_url)
    client = _get_client()

    try:
        repo = client.get_repo(repo_name)
    except GithubException as e:
        raise RuntimeError(f"Could not access repo '{repo_name}': {e.data.get('message', e)}")

    contents = repo.get_contents("")
    file_paths: list[str] = []

    while contents:
        item = contents.pop(0)
        if item.type == "dir":
            if item.name.lower() in _EXCLUDED_DIRS:
                continue  # skip noisy directories entirely, don't even list what's inside
            contents.extend(repo.get_contents(item.path))
        else:
            file_paths.append(item.path)

    return file_paths


def fetch_readme(repo_url: str) -> str:
    """Returns the repo's README content as plain text, or an empty string if missing."""
    repo_name = _repo_name_from_url(repo_url)
    client = _get_client()
    repo = client.get_repo(repo_name)

    try:
        readme = repo.get_readme()
        return readme.decoded_content.decode("utf-8")
    except GithubException:
        return ""  # no README -- the "verify" stage should flag this, not guess


def fetch_file_content(repo_url: str, file_path: str) -> str:
    """Returns the raw text content of one file in the repo."""
    repo_name = _repo_name_from_url(repo_url)
    client = _get_client()
    repo = client.get_repo(repo_name)

    file_content = repo.get_contents(file_path)
    return file_content.decoded_content.decode("utf-8", errors="replace")


def fetch_file_bytes(repo_url: str, file_path: str) -> bytes:
    """Returns the original bytes of one repository file for image/PDF analysis."""
    repo_name = _repo_name_from_url(repo_url)
    client = _get_client()
    repo = client.get_repo(repo_name)

    file_content = repo.get_contents(file_path)
    return file_content.decoded_content


def _repo_name_from_url(repo_url: str) -> str:
    """
    Turns 'https://github.com/owner/repo' into 'owner/repo' for PyGithub.

    Teams commonly paste the URL straight from their browser while
    viewing a specific branch/file/tab, not the bare repo root (e.g.
    '.../owner/repo/tree/main', '.../owner/repo/blob/main/README.md') --
    only the first two path segments after 'github.com/' are ever the
    owner/repo, so everything past that is dropped rather than passed to
    PyGithub as part of the repo name.
    """
    cleaned = repo_url.rstrip("/").removesuffix(".git")
    parts = cleaned.split("github.com/")
    if len(parts) != 2:
        raise ValueError(f"'{repo_url}' doesn't look like a GitHub repo URL.")
    path_parts = parts[1].split("/")
    if len(path_parts) < 2:
        raise ValueError(f"'{repo_url}' doesn't look like a GitHub repo URL.")
    return "/".join(path_parts[:2])


def find_external_resource_links(readme_content: str) -> list[str]:
    """
    Some submissions don't commit a diagram FILE at all -- they link to a
    live board on an external tool from the README instead. Confirmed
    with excalidraw.com in one real submission, but there's no reason a
    different team wouldn't use draw.io, Figma, Miro, Lucidchart, or a
    Google Doc/Drive link instead -- so this checks a broader set of
    common tools, not just the one we happened to see.

    Finds all URLs first, then filters by domain -- rather than anchoring
    the regex to a specific subdomain pattern -- because these tools use
    inconsistent subdomains in practice (e.g. draw.io's actual app lives
    at app.diagrams.net, not www.diagrams.net).

    Fetching and parsing a live external board reliably isn't something
    this project attempts (genuinely less predictable than reading a
    committed file). Returns whatever links were found so gather_node can
    flag them for the judge to check manually, rather than silently
    having no architecture evidence at all.
    """
    all_urls = re.findall(r"https?://\S+", readme_content)
    return [url for url in all_urls if any(domain in url for domain in _EXTERNAL_TOOL_DOMAINS)]
