"""
Runs every example in data/golden_set.json through the compiled LangGraph
pipeline and collects results for scoring.

Kept separate from eval_metrics.py so you can re-run just the (expensive,
LLM-call-heavy) harness once and experiment with scoring logic against the
cached results without re-running the whole pipeline each time.
"""

import json
from pathlib import Path

from graph import compiled_graph

DATA_DIR = Path(__file__).parent / "data"


def load_golden_set() -> list[dict]:
    path = DATA_DIR / "golden_set.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_golden_set.py first.")
    return json.loads(path.read_text())


def run_example(example: dict) -> dict:
    """Runs one golden example through the graph, returns example + actual result."""
    state = {"diff": example["diff"]}
    try:
        result = compiled_graph.invoke(state)
        error = None
    except Exception as e:
        result = {"verified_findings": [], "code_chunks": [], "precedents": [], "final_review": ""}
        error = str(e)

    return {
        "example": example,
        "verified_findings": result.get("verified_findings", []),
        "code_chunks": result.get("code_chunks", []),
        "precedents": result.get("precedents", []),
        "final_review": result.get("final_review", ""),
        "category": result.get("category"),
        "error": error,
    }


def run_harness(golden_set: list[dict] = None, limit: int = None) -> list[dict]:
    """
    Runs the full (or a subset of the) golden set through the pipeline.

    limit: if set, only runs the first N examples — use this to smoke-test
    before committing to a full run, since each example costs several LLM
    calls (specialists + critic + writer) and a full 40-50 example set adds
    up in both time and Groq usage.
    """
    if golden_set is None:
        golden_set = load_golden_set()
    if limit:
        golden_set = golden_set[:limit]

    results = []
    for i, example in enumerate(golden_set, 1):
        print(f"Running example {i}/{len(golden_set)}: {example['source']}")
        result = run_example(example)
        if result["error"]:
            print(f"  ERROR: {result['error']}")
        results.append(result)

    return results