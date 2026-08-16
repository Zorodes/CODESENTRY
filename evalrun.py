"""
Day 4/5 entry point.

Usage:
    python evalrun.py [--limit N]

Runs the golden set through the pipeline, computes finding-level
precision/recall, citation validity rate, and hallucination survival rate, prints
a results table, and saves full per-example results to
eval_results/run_<timestamp>.json so you can track trends across prompt
changes over time.

--limit N: only run the first N golden examples (useful for a quick smoke
test before committing to a full run -- each example costs several LLM calls).

Day 5 additions: Langfuse tracing (each eval run gets a session_id for
grouping in the UI) and wall-clock timing (total + per-example average).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from evalharness import load_golden_set, run_harness
from evalmet import compute_precision_recall, compute_citation_validity_rate, compute_hallucination_survival

RESULTS_DIR = Path(__file__).parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


def score_results(results: list[dict]) -> dict:
    per_example = []
    total_precision = total_recall = total_citation_validity = 0.0
    hallucination_checked = 0
    hallucination_survived = 0
    category_recall = {}  # category -> [recall scores] for "worst category" reporting

    for r in results:
        example = r["example"]
        verified = r["verified_findings"]

        pr = compute_precision_recall(verified, example.get("expected_findings", []))
        citation_validity = compute_citation_validity_rate(
            verified, r["code_chunks"], r["precedents"]
        )

        known_fabricated = example.get("known_fabricated_claims", [])
        hallucinated = None
        if known_fabricated:
            hallucinated = compute_hallucination_survival(verified, known_fabricated)
            hallucination_checked += 1
            if hallucinated:
                hallucination_survived += 1

        total_precision += pr["precision"]
        total_recall += pr["recall"]
        total_citation_validity += citation_validity

        for f in example.get("expected_findings", []):
            cat = f.get("category", "unknown")
            category_recall.setdefault(cat, []).append(pr["recall"])

        per_example.append({
            "source": example["source"],
            "precision": pr["precision"],
            "recall": pr["recall"],
            "citation_validity": citation_validity,
            "hallucination_survived": hallucinated,
            "verified_count": len(verified),
            "expected_count": pr["expected"],
        })

    n = len(results) or 1
    worst_category = None
    if category_recall:
        avg_by_cat = {cat: sum(scores) / len(scores) for cat, scores in category_recall.items()}
        worst_category = min(avg_by_cat.items(), key=lambda x: x[1])

    return {
        "n_examples": len(results),
        "avg_precision": total_precision / n,
        "avg_recall": total_recall / n,
        "avg_citation_validity": total_citation_validity / n,
        "hallucination_rate": (hallucination_survived / hallucination_checked) if hallucination_checked else None,
        "hallucination_checked_count": hallucination_checked,
        "worst_category": worst_category,
        "per_example": per_example,
    }


def print_report(scores: dict, timestamp: str, wall_clock_s: float = None):
    print("=" * 60)
    print(f"CodeSentry Eval Run -- {timestamp}")
    print("=" * 60)
    print(f"Golden set: {scores['n_examples']} examples\n")

    print("Finding-level metrics:")
    print(f"  precision:            {scores['avg_precision']:.2f}")
    print(f"  recall:               {scores['avg_recall']:.2f}")
    print(f"  citation_validity:    {scores['avg_citation_validity']:.2f}")

    if scores["hallucination_rate"] is not None:
        print(f"  hallucination_rate:   {scores['hallucination_rate']:.2f} "
              f"({scores['hallucination_checked_count']} adversarial example(s) checked)")
    else:
        print("  hallucination_rate:   N/A (no examples with known_fabricated_claims)")

    if wall_clock_s is not None:
        n = scores["n_examples"] or 1
        avg_latency = wall_clock_s / n
        print(f"\nTiming:")
        print(f"  total_wall_clock:     {wall_clock_s:.1f}s")
        print(f"  avg_latency/example:  {avg_latency:.1f}s")

    if scores["worst_category"]:
        cat, recall = scores["worst_category"]
        print(f"\nWorst-performing category: {cat} (recall {recall:.2f})")

    print("\nPer-example detail:")
    for ex in scores["per_example"]:
        flag = ""
        if ex["hallucination_survived"] is True:
            flag = "  <- HALLUCINATION SURVIVED"
        print(f"  {ex['source']:35s} P={ex['precision']:.2f} R={ex['recall']:.2f} "
              f"CV={ex['citation_validity']:.2f} verified={ex['verified_count']} expected={ex['expected_count']}{flag}")

    print("=" * 60)


def main():
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    golden_set = load_golden_set()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_id = f"eval_{timestamp}"

    t_start = time.time()
    results = run_harness(golden_set, limit=limit, session_id=session_id)
    wall_clock_s = time.time() - t_start

    scores = score_results(results)
    scores["wall_clock_s"] = round(wall_clock_s, 2)
    scores["avg_latency_s"] = round(wall_clock_s / (len(results) or 1), 2)

    print_report(scores, timestamp, wall_clock_s=wall_clock_s)

    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    out_path.write_text(json.dumps(scores, indent=2, default=str))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()