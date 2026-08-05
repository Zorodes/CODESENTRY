"""
embedder.py - Sentence-transformer wrapper for CodeSentry.

Uses flax-sentence-embeddings/st-codesearch-distilroberta-base:
  - 768-dimensional embeddings
  - Trained on CodeSearchNet: maps code and natural-language queries
    into the same vector space
  - First run downloads ~330MB from HuggingFace, cached in
    ~/.cache/huggingface/ afterwards

Public API:
  embed_one(text)              -> list[float]  (length 768)
  embed_batch(texts, batch_size) -> list[list[float]]

The model is loaded once as a module-level singleton and reused across
all calls in the same process.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
EMBEDDING_DIM = 768

# Module-level singleton - loaded once, reused forever.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(
            f"Loading embedding model ({MODEL_NAME}), "
            "first run downloads ~330MB from HuggingFace. "
            "Subsequent runs load from local cache instantly..."
        )
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")
    return _model


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of strings in batches.

    Args:
        texts:      Non-empty list of strings to embed.
        batch_size: Number of texts processed per forward pass.

    Returns:
        List of 768-dimensional float vectors, one per input string.
    """
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity == dot product post-norm
    )
    return [vec.tolist() for vec in embeddings]


def embed_one(text: str) -> list[float]:
    """
    Embed a single string.

    Returns:
        A 768-dimensional float vector.
    """
    return embed_batch([text], batch_size=1)[0]
