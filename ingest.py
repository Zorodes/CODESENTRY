"""
Day 1 entry point.

Usage:
    python ingest.py <owner> <repo>

Example:
    python ingest.py pydantic pydantic

Fetches the repo's code tree, chunks every file with the AST-aware chunker,
pulls merged PRs + their review comments, and writes everything to
data/code_chunks.json and data/pr_reviews.json.

These two files are the input to Day 2 (embedding + pgvector indexing).
"""

import sys
import json
import time
from pathlib import Path

from github_client import GitHubClient
from chunker import chunk_file

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MAX_FILES_TO_INGEST = 500   # raised for pydantic — larger, well-organized codebase,
                             # worth deeper coverage than FastAPI's 300-file cap
MAX_PRS_TO_INGEST = 100     # raised since pydantic has much richer human review history
                             # than FastAPI — worth pulling more real examples for the
                             # golden set now that they're actually available


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
    prs = client.get_closed_prs(max_pages=15)[:MAX_PRS_TO_INGEST]
    print(f"Found {len(prs)} merged, human-authored PRs (bot PRs like Dependabot filtered out).")

    enriched = []
    for i, pr in enumerate(prs, 1):
        try:
            comments = client.get_pr_review_comments(pr["number"])
            reviews = client.get_pr_reviews(pr["number"])
            diff = client.get_pr_diff(pr["number"])
        except Exception as e:
            print(f"  skip PR #{pr['number']}: {e}")
            continue

        enriched.append({**pr, "review_comments": comments, "reviews": reviews, "diff": diff})

        if i % 10 == 0:
            print(f"  processed {i}/{len(prs)} PRs")
        time.sleep(0.1)

    return enriched


def main():
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <owner> <repo> [--prs-only]")
        sys.exit(1)

    owner, repo = sys.argv[1], sys.argv[2]
    prs_only = "--prs-only" in sys.argv
    client = GitHubClient(owner, repo)

    if not prs_only:
        chunks = ingest_codebase(client)
        chunks_path = DATA_DIR / "code_chunks.json"
        chunks_path.write_text(json.dumps(chunks, indent=2))
        print(f"Wrote {len(chunks)} chunks to {chunks_path}")
    else:
        print("--prs-only set, skipping code chunk ingestion (code_chunks.json untouched)")

    prs = ingest_pr_history(client)
    prs_path = DATA_DIR / "pr_reviews.json"
    prs_path.write_text(json.dumps(prs, indent=2))
    print(f"Wrote {len(prs)} PRs with review history to {prs_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()