"""
db.py - Postgres / pgvector persistence layer for CodeSentry Day 2.

Responsibilities:
  - Manage the psycopg2 connection pool (DATABASE_URL from env).
  - Define and migrate the schema: code_chunks and pr_precedents tables,
    HNSW indexes for vector similarity, GIN indexes for full-text search.
  - Batch-insert code chunks and PR precedents.
  - Hybrid search (vector cosine + Postgres full-text) merged with
    Reciprocal Rank Fusion (RRF, k=60).
"""

import os
import json
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Module-level connection pool, initialised lazily.
_pool: pg_pool.ThreadedConnectionPool | None = None

# RRF constant - larger k de-emphasises rank differences.
RRF_K = 60


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to your .env file. "
                "Example: DATABASE_URL=postgresql://user:pass@localhost:5432/codesentry"
            )
        _pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
    return _pool


def get_conn() -> psycopg2.extensions.connection:
    """Borrow a connection from the pool. Caller must call put_conn()."""
    return _get_pool().getconn()


def put_conn(conn: psycopg2.extensions.connection) -> None:
    """Return a connection to the pool."""
    _get_pool().putconn(conn)


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS code_chunks (
    id            SERIAL PRIMARY KEY,
    file_path     TEXT    NOT NULL,
    language      TEXT    NOT NULL,
    chunk_type    TEXT    NOT NULL,
    name          TEXT    NOT NULL,
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    code          TEXT    NOT NULL,
    embedding     vector(768),
    fts           tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(name, '') || ' ' || coalesce(code, ''))
                  ) STORED
);

CREATE TABLE IF NOT EXISTS pr_precedents (
    id            SERIAL PRIMARY KEY,
    pr_number     INTEGER NOT NULL,
    pr_title      TEXT    NOT NULL,
    file_path     TEXT    NOT NULL,
    diff_hunk     TEXT    NOT NULL,
    comment_body  TEXT    NOT NULL,
    embedding     vector(768),
    fts           tsvector GENERATED ALWAYS AS (
                    to_tsvector('english',
                        coalesce(pr_title, '') || ' ' ||
                        coalesce(file_path, '') || ' ' ||
                        coalesce(diff_hunk, '') || ' ' ||
                        coalesce(comment_body, ''))
                  ) STORED
);

-- HNSW indexes for approximate nearest-neighbour vector search.
CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding
    ON code_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_pr_precedents_embedding
    ON pr_precedents USING hnsw (embedding vector_cosine_ops);

-- GIN indexes for full-text keyword search.
CREATE INDEX IF NOT EXISTS idx_code_chunks_fts
    ON code_chunks USING gin (fts);

CREATE INDEX IF NOT EXISTS idx_pr_precedents_fts
    ON pr_precedents USING gin (fts);
"""

_RESET_SQL = """
DROP TABLE IF EXISTS pr_precedents CASCADE;
DROP TABLE IF EXISTS code_chunks CASCADE;
"""


def init_schema() -> None:
    """Create tables and indexes if they do not already exist."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
        print("Schema initialised (tables and indexes are up to date).")
    finally:
        put_conn(conn)


def reset_schema() -> None:
    """Drop all CodeSentry tables then recreate them from scratch."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_RESET_SQL)
        conn.commit()
        print("Existing tables dropped.")
    finally:
        put_conn(conn)
    init_schema()


# ---------------------------------------------------------------------------
# Batch insert helpers
# ---------------------------------------------------------------------------

def insert_code_chunks(chunks: list[dict[str, Any]]) -> None:
    """
    Insert a batch of code chunk dicts.  Each dict must have the keys
    produced by CodeChunk.to_dict() plus an 'embedding' key (list[float]).
    """
    if not chunks:
        return

    rows = [
        (
            c["file_path"],
            c["language"],
            c["chunk_type"],
            c["name"],
            c["start_line"],
            c["end_line"],
            c["code"],
            c["embedding"],          # list[float] -> psycopg2 will serialise
        )
        for c in chunks
    ]

    sql = """
        INSERT INTO code_chunks
            (file_path, language, chunk_type, name, start_line, end_line, code, embedding)
        VALUES %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, sql, rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s::vector)",
                page_size=100,
            )
        conn.commit()
    finally:
        put_conn(conn)


def insert_pr_precedents(precedents: list[dict[str, Any]]) -> None:
    """
    Insert a batch of PR precedent dicts.  Each dict must have:
        pr_number, pr_title, file_path, diff_hunk, comment_body, embedding.
    """
    if not precedents:
        return

    rows = [
        (
            p["pr_number"],
            p["pr_title"],
            p["file_path"],
            p["diff_hunk"],
            p["comment_body"],
            p["embedding"],
        )
        for p in precedents
    ]

    sql = """
        INSERT INTO pr_precedents
            (pr_number, pr_title, file_path, diff_hunk, comment_body, embedding)
        VALUES %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, sql, rows,
                template="(%s, %s, %s, %s, %s, %s::vector)",
                page_size=100,
            )
        conn.commit()
    finally:
        put_conn(conn)


# ---------------------------------------------------------------------------
# Row counts (for /index-status endpoint)
# ---------------------------------------------------------------------------

def get_table_counts() -> dict[str, int]:
    """Return row counts for both tables.  Returns -1 if DB is not reachable."""
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM code_chunks;")
                code_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM pr_precedents;")
                pr_count = cur.fetchone()[0]
            return {"code_chunks": code_count, "pr_precedents": pr_count}
        finally:
            put_conn(conn)
    except Exception as exc:
        raise RuntimeError(f"Database not reachable: {exc}") from exc


# ---------------------------------------------------------------------------
# Hybrid search: vector + full-text, fused with RRF
# ---------------------------------------------------------------------------

def _rrf_merge(
    vector_rows: list[dict],
    fts_rows: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    Both lists are already ranked (index 0 = best match) and should contain the
    full row data necessary to return a merged result set without dropping any
    FTS-only matches.
    """
    vector_rank: dict[int, int] = {
        row["id"]: rank + 1 for rank, row in enumerate(vector_rows)
    }
    fts_rank: dict[int, int] = {
        row["id"]: rank + 1 for rank, row in enumerate(fts_rows)
    }

    all_ids = set(vector_rank) | set(fts_rank)
    rrf_scores: dict[int, float] = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in vector_rank:
            score += 1.0 / (RRF_K + vector_rank[doc_id])
        if doc_id in fts_rank:
            score += 1.0 / (RRF_K + fts_rank[doc_id])
        rrf_scores[doc_id] = score

    row_lookup: dict[int, dict] = {}
    for row in vector_rows:
        row_lookup[row["id"]] = row
    for row in fts_rows:
        row_lookup.setdefault(row["id"], row)

    ranked_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
    result = []
    for doc_id in ranked_ids[:top_k]:
        if doc_id in row_lookup:
            result.append(row_lookup[doc_id])
    return result


def hybrid_search_code(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 10,
    fts_candidates: int = 50,
) -> list[dict[str, Any]]:
    """
    Hybrid search over code_chunks.

    1. Vector arm: cosine similarity via pgvector (top fts_candidates rows).
    2. FTS arm: Postgres full-text search via ts_rank (top fts_candidates rows).
    3. Merge with RRF and return top_k results.

    Returns a list of dicts with all chunk columns plus 'rrf_score'.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # --- Vector arm ---
            cur.execute(
                """
                SELECT id, file_path, language, chunk_type, name,
                       start_line, end_line, code,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   code_chunks
                WHERE  embedding IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (query_embedding, query_embedding, fts_candidates),
            )
            vector_rows = cur.fetchall()

            # --- FTS arm (plainto_tsquery handles arbitrary text safely) ---
            cur.execute(
                """
                SELECT id, file_path, language, chunk_type, name,
                       start_line, end_line, code,
                       ts_rank(fts, plainto_tsquery('english', %s)) AS rank
                FROM   code_chunks
                WHERE  fts @@ plainto_tsquery('english', %s)
                ORDER  BY rank DESC
                LIMIT  %s
                """,
                (query_text, query_text, fts_candidates),
            )
            fts_rows = cur.fetchall()

    finally:
        put_conn(conn)

    vector_rows = [dict(r) for r in vector_rows]
    fts_rows    = [dict(r) for r in fts_rows]

    merged = _rrf_merge(vector_rows, fts_rows, top_k)

    # Annotate with rrf position.
    for rank, row in enumerate(merged, 1):
        row["rrf_rank"] = rank

    return merged


def hybrid_search_precedents(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 10,
    fts_candidates: int = 50,
) -> list[dict[str, Any]]:
    """
    Hybrid search over pr_precedents. Same RRF strategy as hybrid_search_code.

    Returns a list of dicts with all precedent columns plus 'rrf_rank'.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # --- Vector arm ---
            cur.execute(
                """
                SELECT id, pr_number, pr_title, file_path,
                       diff_hunk, comment_body,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   pr_precedents
                WHERE  embedding IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (query_embedding, query_embedding, fts_candidates),
            )
            vector_rows = cur.fetchall()

            # --- FTS arm (plainto_tsquery handles arbitrary text safely) ---
            cur.execute(
                """
                SELECT id, pr_number, pr_title, file_path,
                       diff_hunk, comment_body,
                       ts_rank(fts, plainto_tsquery('english', %s)) AS rank
                FROM   pr_precedents
                WHERE  fts @@ plainto_tsquery('english', %s)
                ORDER  BY rank DESC
                LIMIT  %s
                """,
                (query_text, query_text, fts_candidates),
            )
            fts_rows = cur.fetchall()

    finally:
        put_conn(conn)

    vector_rows = [dict(r) for r in vector_rows]
    fts_rows    = [dict(r) for r in fts_rows]

    merged = _rrf_merge(vector_rows, fts_rows, top_k)

    for rank, row in enumerate(merged, 1):
        row["rrf_rank"] = rank

    return merged