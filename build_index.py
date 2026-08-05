"""
build_index.py - CLI tool to embed code chunks and PR precedents and load
them into Postgres/pgvector.

Usage:
    python build_index.py            # append new rows (safe to re-run)
    python build_index.py --reset    # drop + recreate tables first

The script reads:
    data/code_chunks.json   - produced by Day 1 ingest.py
    data/pr_reviews.json    - produced by Day 1 ingest.py

And writes embedded rows into:
    code_chunks     table (Postgres)
    pr_precedents   table (Postgres)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import db
import embedder

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Text preparation helpers
# ---------------------------------------------------------------------------

def _chunk_text(chunk: dict) -> str:
    """
    Build the string that will be embedded for a code chunk.

    Prefixing with file path and name gives the model just enough structural
    context to distinguish e.g. two functions that have identical code but live
    in completely different modules.
    """
    return f"{chunk['file_path']} :: {chunk['name']}\n{chunk['code']}"


def _precedent_text(p: dict) -> str:
    """
    Build the string that will be embedded for a PR precedent row.
    """
    return (
        f"PR #{p['pr_number']} - {p['pr_title']}\n"
        f"File: {p['file_path']}\n"
        f"Diff:\n{p['diff_hunk']}\n"
        f"Review comment: {p['comment_body']}"
    )


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def index_code_chunks(chunks: list[dict]) -> None:
    total = len(chunks)
    print(f"Embedding {total} code chunks (batch size {BATCH_SIZE})...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [_chunk_text(c) for c in batch]

        t0 = time.time()
        embeddings = embedder.embed_batch(texts, batch_size=BATCH_SIZE)
        elapsed = time.time() - t0

        # Attach embeddings to the chunk dicts before inserting.
        enriched = []
        for chunk, emb in zip(batch, embeddings):
            enriched.append({**chunk, "embedding": emb})

        db.insert_code_chunks(enriched)

        batch_end = min(batch_start + BATCH_SIZE, total)
        print(
            f"  code_chunks [{batch_end}/{total}] "
            f"embedded in {elapsed:.1f}s, inserted OK"
        )

    print(f"Done: {total} code chunks indexed.")


def index_pr_precedents(pr_data: list[dict]) -> None:
    """
    Flatten PR + review_comments into individual precedent rows, then embed.
    Each review comment on a PR becomes one row in pr_precedents.
    PRs with no review comments are skipped (no comment_body to store).
    """
    # Flatten
    precedents: list[dict] = []
    for pr in pr_data:
        comments = pr.get("review_comments", [])
        if not comments:
            continue
        for comment in comments:
            precedents.append({
                "pr_number":   pr["number"],
                "pr_title":    pr.get("title", ""),
                "file_path":   comment.get("path", ""),
                "diff_hunk":   comment.get("diff_hunk", ""),
                "comment_body": comment.get("body", ""),
            })

    total = len(precedents)
    if total == 0:
        print("No PR review comments found - skipping pr_precedents indexing.")
        return

    print(f"Embedding {total} PR precedent rows (batch size {BATCH_SIZE})...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = precedents[batch_start : batch_start + BATCH_SIZE]
        texts = [_precedent_text(p) for p in batch]

        t0 = time.time()
        embeddings = embedder.embed_batch(texts, batch_size=BATCH_SIZE)
        elapsed = time.time() - t0

        enriched = []
        for prec, emb in zip(batch, embeddings):
            enriched.append({**prec, "embedding": emb})

        db.insert_pr_precedents(enriched)

        batch_end = min(batch_start + BATCH_SIZE, total)
        print(
            f"  pr_precedents [{batch_end}/{total}] "
            f"embedded in {elapsed:.1f}s, inserted OK"
        )

    print(f"Done: {total} PR precedent rows indexed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pgvector index for CodeSentry.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables before indexing.",
    )
    args = parser.parse_args()

    # Load data files.
    chunks_path = DATA_DIR / "code_chunks.json"
    prs_path    = DATA_DIR / "pr_reviews.json"

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)
    if not prs_path.exists():
        print(f"ERROR: {prs_path} not found. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {chunks_path}...")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Reading {prs_path}...")
    pr_data = json.loads(prs_path.read_text(encoding="utf-8"))

    # Schema setup.
    if args.reset:
        print("--reset flag set: dropping existing tables...")
        db.reset_schema()
    else:
        db.init_schema()

    # Index both collections.
    index_code_chunks(chunks)
    index_pr_precedents(pr_data)

    # Final counts.
    counts = db.get_table_counts()
    print(
        f"\nIndex build complete.\n"
        f"  code_chunks   rows: {counts['code_chunks']}\n"
        f"  pr_precedents rows: {counts['pr_precedents']}"
    )


if __name__ == "__main__":
    main()
