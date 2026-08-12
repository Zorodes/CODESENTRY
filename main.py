"""
main.py - FastAPI application for CodeSentry.

Endpoints:
  POST /review
      Body: { "diff": str, "repo_owner": str, "repo_name": str }
      Returns: { "review": str, "code_chunks_retrieved": list, "precedents_retrieved": list }

  GET /index-status
      Returns: { "code_chunks": int, "pr_precedents": int }

Run locally:
    uvicorn main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

import db
from graph import compiled_graph

app = FastAPI(
    title="CodeSentry",
    description="Multi-agent PR review system powered by pgvector + Groq.",
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    diff: str
    repo_owner: str = ""
    repo_name: str  = ""

    @field_validator("diff")
    @classmethod
    def diff_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("diff must not be empty")
        return v


class ChunkResult(BaseModel):
    id: int
    file_path: str
    language: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    code: str
    similarity: float | None = None
    rrf_rank: int | None = None


class PrecedentResult(BaseModel):
    id: int
    pr_number: int
    pr_title: str
    file_path: str
    diff_hunk: str
    comment_body: str
    similarity: float | None = None
    rrf_rank: int | None = None


class ReviewResponse(BaseModel):
    review: str
    code_chunks_retrieved: list[ChunkResult]
    precedents_retrieved: list[PrecedentResult]


class IndexStatusResponse(BaseModel):
    code_chunks: int
    pr_precedents: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/review", response_model=ReviewResponse)
def post_review(request: ReviewRequest) -> ReviewResponse:
    """
    Embed the diff, run hybrid retrieval, call Groq, return the review.
    """
    # Validate diff (also caught by Pydantic, but explicit error is clearer).
    if not request.diff.strip():
        raise HTTPException(status_code=422, detail="diff must not be empty")

    # Check DB connectivity before doing expensive embedding work.
    try:
        db.get_table_counts()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database not ready: {exc}",
        ) from exc

    try:
        state_input = {"diff": request.diff}
        final_state = compiled_graph.invoke(state_input)
        result = {
            "review": final_state.get("final_review", ""),
            "code_chunks_retrieved": final_state.get("code_chunks", []),
            "precedents_retrieved": final_state.get("precedents", [])
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Review failed: {exc}"
        ) from exc

    return ReviewResponse(
        review=result["review"],
        code_chunks_retrieved=[
            ChunkResult(**c) for c in result["code_chunks_retrieved"]
        ],
        precedents_retrieved=[
            PrecedentResult(**p) for p in result["precedents_retrieved"]
        ],
    )


@app.get("/index-status", response_model=IndexStatusResponse)
def get_index_status() -> IndexStatusResponse:
    """
    Return row counts for both indexed tables.
    Returns 503 if the database is not reachable or tables do not exist yet.
    """
    try:
        counts = db.get_table_counts()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database not ready: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database error: {exc}",
        ) from exc

    return IndexStatusResponse(
        code_chunks=counts["code_chunks"],
        pr_precedents=counts["pr_precedents"],
    )


# ---------------------------------------------------------------------------
# Health probe (bonus, simple)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
