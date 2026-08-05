"""
retrieve_agent.py - Hybrid retrieval + Groq LLM review for CodeSentry.

Entry point: review_diff(diff_text) -> dict

Steps:
  1. Embed the diff with the code-search model.
  2. Run hybrid_search_code and hybrid_search_precedents (vector + FTS, RRF-merged).
  3. Assemble a numbered context block with citation IDs [code:N] / [precedent:N].
  4. Call Groq (llama-3.3-70b-versatile) with a strict system prompt that
     demands citations and forbids hallucinating line numbers or PR numbers.
  5. Return the review text plus the raw retrieved rows.

Output sections in the review:
  - Bug Risk
  - Convention Consistency
  - Test Coverage
"""

from __future__ import annotations

import os

from groq import Groq
from dotenv import load_dotenv

import db
import embedder

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# Number of items retrieved from each arm.
CODE_TOP_K      = 8
PRECEDENT_TOP_K = 5

# Groq client - lazily initialised.
_groq_client: Groq | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _build_code_context(code_chunks: list[dict]) -> str:
    lines = []
    for chunk in code_chunks:
        cid = f"[code:{chunk['id']}]"
        lines.append(
            f"{cid} {chunk['file_path']} lines {chunk['start_line']}-{chunk['end_line']} "
            f"({chunk['language']} {chunk['chunk_type']}: {chunk['name']})\n"
            f"```{chunk['language']}\n{chunk['code']}\n```"
        )
    return "\n\n".join(lines)


def _build_precedent_context(precedents: list[dict]) -> str:
    lines = []
    for p in precedents:
        pid = f"[precedent:{p['id']}]"
        lines.append(
            f"{pid} PR #{p['pr_number']} - {p['pr_title']}\n"
            f"File: {p['file_path']}\n"
            f"Diff hunk:\n```diff\n{p['diff_hunk']}\n```\n"
            f"Reviewer comment: {p['comment_body']}"
        )
    return "\n\n".join(lines)


_SYSTEM_PROMPT = """\
You are CodeSentry, an expert automated code reviewer. You are given:
  1. A git diff from a pull request under review.
  2. A set of retrieved CODE CHUNKS from the codebase, each labelled [code:N].
  3. A set of retrieved PR PRECEDENTS (past review comments), each labelled [precedent:N].

Your job is to produce a structured code review with exactly three sections:

## Bug Risk
Identify any bugs, logic errors, security issues, or race conditions.

## Convention Consistency
Flag style or convention deviations relative to the retrieved code context.

## Test Coverage
Comment on whether the diff is adequately tested.

Rules you must follow without exception:
- Every factual claim MUST be backed by a citation in the form [code:N] or [precedent:N].
- Never invent line numbers, function names, or PR numbers that are not present in the retrieved context.
- If the retrieved context does not contain enough information to support a claim, say "Insufficient context to assess."
- Do not output any text outside the three sections above.
- Be concise but precise. Each section should be 2-5 bullet points.
"""


# ---------------------------------------------------------------------------
# Main review function
# ---------------------------------------------------------------------------

def review_diff(diff_text: str) -> dict:
    """
    Embed the diff, retrieve context, call Groq, return structured result.

    Returns:
        {
            "review": str,
            "code_chunks_retrieved": list[dict],
            "precedents_retrieved":  list[dict],
        }
    """
    if not diff_text or not diff_text.strip():
        raise ValueError("diff_text is empty - nothing to review.")

    # 1. Embed the diff.
    query_embedding = embedder.embed_one(diff_text[:4096])  # guard against huge diffs

    # 2. Retrieve from both tables.
    code_chunks = db.hybrid_search_code(
        query_embedding=query_embedding,
        query_text=diff_text,
        top_k=CODE_TOP_K,
    )
    precedents = db.hybrid_search_precedents(
        query_embedding=query_embedding,
        query_text=diff_text,
        top_k=PRECEDENT_TOP_K,
    )

    # 3. Build the user message.
    code_ctx      = _build_code_context(code_chunks)
    precedent_ctx = _build_precedent_context(precedents)

    user_message = (
        f"## Diff under review\n```diff\n{diff_text}\n```\n\n"
        f"## Retrieved code chunks\n{code_ctx or 'None retrieved.'}\n\n"
        f"## Retrieved PR precedents\n{precedent_ctx or 'None retrieved.'}"
    )

    # 4. Call Groq.
    client = _get_groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,   # low temperature for consistent structured output
        max_tokens=2048,
    )

    review_text = response.choices[0].message.content

    # 5. Return everything the API layer needs.
    # Strip embedding vectors from the returned rows (large, not useful to callers).
    def _strip(row: dict) -> dict:
        return {k: v for k, v in row.items() if k != "embedding"}

    return {
        "review":                  review_text,
        "code_chunks_retrieved":   [_strip(c) for c in code_chunks],
        "precedents_retrieved":    [_strip(p) for p in precedents],
    }
