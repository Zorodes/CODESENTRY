"""
Day 1 entry point.

Usage:
    python ingest.py <owner> <repo>

Example:
    python ingest.py tiangolo fastapi

Fetches the repo's code tree, chunks every file with the AST-aware chunker,
pulls merged PRs + their review comments, and writes everything to
data/code_chunks.json and data/pr_reviews.json.

These two files are the input to Day 2 (embedding + pgvector indexing).
Keep repo choice modest at first (a few hundred files) — you can always
re-run against a bigger repo once the pipeline is proven.
"""

import sys
import json
import time
from pathlib import Path

from github_client import GitHubClient
from chunker import chunk_file

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MAX_FILES_TO_INGEST = 300   # cap for a first run; raise once pipeline is proven
MAX_PRS_TO_INGEST = 40      # this becomes your eval golden-set candidate pool


def ingest_codebase(client: GitHubClient) -> list[dict]:
    print("Fetching repo tree...")
    tree = client.get_repo_tree()
    print(f"Found {len(tree)} code files (post-filter).")

    files_to_process = tree[:MAX_FILES_TO_INGEST]
    all_chunks = []

    for i, item in enumerate(files_to_process, 1):
        path = item["path"]
        try:
            content = client.get_file_content(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue

        chunks = chunk_file(path, content)
        all_chunks.extend(c.to_dict() for c in chunks)

        if i % 25 == 0:
            print(f"  processed {i}/{len(files_to_process)} files, {len(all_chunks)} chunks so far")
        time.sleep(0.05)  # stay comfortably under rate limits

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def ingest_pr_history(client: GitHubClient) -> list[dict]:
    print("Fetching merged PRs...")
    prs = client.get_closed_prs(max_pages=3)[:MAX_PRS_TO_INGEST]
    print(f"Found {len(prs)} merged PRs.")

    enriched = []
    for i, pr in enumerate(prs, 1):
        try:
            comments = client.get_pr_review_comments(pr["number"])
            diff = client.get_pr_diff(pr["number"])
        except Exception as e:
            print(f"  skip PR #{pr['number']}: {e}")
            continue

        enriched.append({**pr, "review_comments": comments, "diff": diff})

        if i % 10 == 0:
            print(f"  processed {i}/{len(prs)} PRs")
        time.sleep(0.1)

    return enriched


def main():
    if len(sys.argv) != 3:
        print("Usage: python ingest.py <owner> <repo>")
        sys.exit(1)

    owner, repo = sys.argv[1], sys.argv[2]
    client = GitHubClient(owner, repo)

    chunks = ingest_codebase(client)
    chunks_path = DATA_DIR / "code_chunks.json"
    chunks_path.write_text(json.dumps(chunks, indent=2))
    print(f"Wrote {len(chunks)} chunks to {chunks_path}")

    prs = ingest_pr_history(client)
    prs_path = DATA_DIR / "pr_reviews.json"
    prs_path.write_text(json.dumps(prs, indent=2))
    print(f"Wrote {len(prs)} PRs with review history to {prs_path}")

    print("\nDay 1 done. Next: embed code_chunks.json into pgvector (Day 2).")


if __name__ == "__main__":
    main()
