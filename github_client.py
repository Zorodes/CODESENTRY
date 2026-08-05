"""
GitHub API client for CodeSentry ingestion.

Pulls:
- Full repo file tree (filtered to code files)
- File contents at a given ref
- Commit history
- Closed PRs with their review comments (used later as the golden eval set
  and as retrieval precedent for "how does this team usually review X")
"""

import base64
import time
import requests

from config import GITHUB_API_BASE, HEADERS, CODE_EXTENSIONS, SKIP_DIRS, MAX_FILE_SIZE_BYTES


class GitHubClient:
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, url: str, params: dict = None) -> requests.Response:
        resp = self.session.get(url, params=params)
        if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers:
            if resp.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                print(f"Rate limited. Sleeping {wait}s.")
                time.sleep(wait)
                return self._get(url, params)
        resp.raise_for_status()
        return resp

    def _paginate(self, url: str, params: dict = None, max_pages: int = 10) -> list:
        params = dict(params or {})
        params.setdefault("per_page", 100)
        results = []
        page = 1
        while page <= max_pages:
            params["page"] = page
            resp = self._get(url, params)
            batch = resp.json()
            if not batch:
                break
            results.extend(batch)
            if "next" not in resp.links:
                break
            page += 1
        return results

    def get_default_branch(self) -> str:
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}"
        return self._get(url).json()["default_branch"]

    def get_repo_tree(self, ref: str = None) -> list[dict]:
        """Returns filtered list of {path, sha, size} for code files only."""
        ref = ref or self.get_default_branch()
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/git/trees/{ref}"
        resp = self._get(url, params={"recursive": "1"})
        tree = resp.json().get("tree", [])

        filtered = []
        for item in tree:
            if item["type"] != "blob":
                continue
            path = item["path"]
            if any(f"/{skip}/" in f"/{path}/" for skip in SKIP_DIRS):
                continue
            if not any(path.endswith(ext) for ext in CODE_EXTENSIONS):
                continue
            if item.get("size", 0) > MAX_FILE_SIZE_BYTES:
                continue
            filtered.append({"path": path, "sha": item["sha"], "size": item.get("size", 0)})
        return filtered

    def get_file_content(self, path: str, ref: str = None) -> str:
        ref = ref or self.get_default_branch()
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/contents/{path}"
        resp = self._get(url, params={"ref": ref})
        data = resp.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return data.get("content", "")

    def get_commits(self, max_pages: int = 5) -> list[dict]:
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/commits"
        raw = self._paginate(url, max_pages=max_pages)
        return [
            {
                "sha": c["sha"],
                "message": c["commit"]["message"],
                "author": (c["commit"]["author"] or {}).get("name"),
                "date": (c["commit"]["author"] or {}).get("date"),
            }
            for c in raw
        ]

    def get_closed_prs(self, max_pages: int = 3) -> list[dict]:
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls"
        raw = self._paginate(
            url, params={"state": "closed", "sort": "updated", "direction": "desc"},
            max_pages=max_pages,
        )
        return [
            {
                "number": pr["number"],
                "title": pr["title"],
                "body": pr.get("body") or "",
                "merged_at": pr.get("merged_at"),
                "base_sha": pr["base"]["sha"],
                "head_sha": pr["head"]["sha"],
            }
            for pr in raw
            if pr.get("merged_at")  # only merged PRs are useful precedent
        ]

    def get_pr_review_comments(self, pr_number: int) -> list[dict]:
        """Line-level review comments on a specific PR (the actual review feedback)."""
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments"
        raw = self._paginate(url, max_pages=3)
        return [
            {
                "path": c.get("path"),
                "line": c.get("line"),
                "body": c["body"],
                "author": c["user"]["login"] if c.get("user") else None,
                "diff_hunk": c.get("diff_hunk"),
            }
            for c in raw
        ]

    def get_pr_diff(self, pr_number: int) -> str:
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        headers = {**self.session.headers, "Accept": "application/vnd.github.v3.diff"}
        resp = self.session.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text
