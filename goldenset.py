"""
Builds data/golden_set.json from two sources:

1. Real PRs (data/pr_reviews.json) — for each merged PR that has review
   comments, this pulls out the diff + comment text as a candidate example.
   You don't need the code to be "buggy" — whatever the human reviewer
   actually said (a bug, a style nitpick, a missing test, even nothing)
   becomes the ground truth for that example.

2. Adversarial examples — hand-written diffs designed to probe specific
   failure modes, the same pattern as your unsafe_query + "turbo
   encabulator" test. A handful of starters are included below; add more
   directly in the ADVERSARIAL_EXAMPLES list.

This script does NOT try to fully automate labeling — it extracts
candidates and does lightweight auto-categorization of the reviewer
comment (bug/convention/test/other) using keyword heuristics, but you
should skim the output and fix any obviously wrong categorizations before
using it for real eval numbers. That skim is quick (a few minutes for ~30
examples) and matters more for the quality of your metrics than anything
else in Day 4.

Usage:
    python build_golden_set.py
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MAX_REAL_EXAMPLES = 30

# --- lightweight heuristics to pre-categorize real reviewer comments ---
# This is just a starting guess to save you typing — always spot-check it.

BUG_KEYWORDS = ["bug", "error", "exception", "crash", "fail", "incorrect", "wrong",
                "vulnerab", "injection", "race condition", "null", "none check", "edge case"]
CONVENTION_KEYWORDS = ["naming", "style", "convention", "pep8", "pep 8", "format",
                       "consistent", "should use", "prefer", "type hint", "typing"]
TEST_KEYWORDS = ["test", "coverage", "assert", "unit test", "missing test"]


def guess_category(comment_body: str) -> str:
    text = comment_body.lower()
    if any(kw in text for kw in TEST_KEYWORDS):
        return "test_coverage"
    if any(kw in text for kw in BUG_KEYWORDS):
        return "bug_risk"
    if any(kw in text for kw in CONVENTION_KEYWORDS):
        return "convention"
    return "other"  # you should manually re-check these


def build_real_examples() -> list[dict]:
    path = DATA_DIR / "pr_reviews.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — this comes from Day 1's ingest.py")

    prs = json.loads(path.read_text())
    examples = []

    for pr in prs:
        comments = pr.get("review_comments", [])
        reviews = pr.get("reviews", [])
        diff = pr.get("diff", "")

        if not diff or (not comments and not reviews):
            continue  # need both a diff and at least some real feedback to be useful

        expected_findings = []
        for c in comments:
            body = c.get("body", "").strip()
            if len(body) < 10:
                continue  # skip trivial/empty comments ("lgtm", "+1", etc.)
            expected_findings.append({
                "category": guess_category(body),
                "should_flag": True,
                "description": body[:300],  # trim very long comments
            })

        for r in reviews:
            body = r.get("body", "").strip()
            # skip bare approvals with no real feedback text
            if len(body) < 10 or body.lower().strip() in ("lgtm", "looks good", "approved"):
                continue
            expected_findings.append({
                "category": guess_category(body),
                "should_flag": True,
                "description": body[:300],
            })

        if not expected_findings:
            continue

        examples.append({
            "diff": diff,
            "expected_findings": expected_findings,
            "source": f"real_pr_{pr.get('number')}",
            "notes": f'PR title: "{pr.get("title", "")}" — auto-categorized, VERIFY before trusting metrics',
        })

        if len(examples) >= MAX_REAL_EXAMPLES:
            break

    return examples


# --- adversarial examples: hand-written, testing specific failure modes ---
# Add more of these directly. Each one should target ONE specific thing
# you're worried the pipeline might get wrong.

ADVERSARIAL_EXAMPLES = [
    {
        "diff": """diff --git a/main.py b/main.py
index 1234567..890abcd 100644
--- a/main.py
+++ b/main.py
@@ -150,3 +150,5 @@ def health() -> dict:
     return {"status": "ok"}
+
+# Introducing an imaginary sql injection vulnerability and turbo encabulator issue.
+def unsafe_query(user_id):
+    db.execute(f"SELECT * FROM users WHERE id = {user_id}")
""",
        "expected_findings": [
            {"category": "bug_risk", "should_flag": True,
             "description": "SQL injection via f-string interpolation in unsafe_query"},
        ],
        "source": "adversarial_hallucination_probe",
        "known_fabricated_claims": ["turbo encabulator", "encabulator"],
        "notes": "Real bug + fake claim planted only in a comment. Must catch the real bug "
                 "AND must NOT flag 'turbo encabulator' as if it were real. This is your "
                 "primary hallucination_rate test case.",
    },
    {
        "diff": """diff --git a/utils.py b/utils.py
index abc123..def456 100644
--- a/utils.py
+++ b/utils.py
@@ -10,3 +10,6 @@ def format_currency(amount: float) -> str:
     return f"${amount:,.2f}"
+
+def format_percentage(value: float) -> str:
+    return f"{value:.1%}"
""",
        "expected_findings": [],
        "source": "adversarial_clean_diff",
        "notes": "A genuinely clean, small, well-written addition. Correct answer is ZERO "
                 "findings. Tests whether the pipeline forces findings when there's nothing "
                 "real to say — your false-positive-rate test case.",
    },
    {
        "diff": """diff --git a/payments/processor.py b/payments/processor.py
index 111222..333444 100644
--- a/payments/processor.py
+++ b/payments/processor.py
@@ -40,3 +40,8 @@ def process_payment(order):
     return charge_result
+
+def refund_payment(order_id, amount):
+    stripe_client.refunds.create(payment_intent=order_id, amount=amount)
+    order = get_order(order_id)
+    order.status = "refunded"
""",
        "expected_findings": [
            {"category": "bug_risk", "should_flag": True,
             "description": "no error handling if the Stripe refund call fails, order status "
                             "still gets marked 'refunded' regardless of API call outcome"},
            {"category": "test_coverage", "should_flag": True,
             "description": "no tests for the new refund_payment function"},
        ],
        "source": "adversarial_new_code_no_precedent",
        "notes": "Brand-new function in a file/domain likely NOT well represented in your "
                 "index. Tests whether [diff]-grounded citation works when retrieval returns "
                 "weak/irrelevant context.",
    },
    {
        "diff": """diff --git a/README.md b/README.md
index 999888..777666 100644
--- a/README.md
+++ b/README.md
@@ -5,3 +5,4 @@
 Run `pip install -r requirements.txt` to get started.
+See CONTRIBUTING.md for development setup instructions.
""",
        "expected_findings": [],
        "source": "adversarial_non_code_diff",
        "notes": "Pure documentation change, no code at all. Correct answer is ZERO findings "
                 "across all categories. Tests whether specialists correctly recognize "
                 "'nothing to review here' rather than manufacturing a finding about a "
                 "markdown file.",
    },
]


def main():
    print("Building golden set...")

    real_examples = build_real_examples()
    print(f"Extracted {len(real_examples)} candidate examples from real PR history.")
    print("NOTE: category guesses are heuristic — spot-check data/golden_set.json before trusting eval numbers.")

    all_examples = real_examples + ADVERSARIAL_EXAMPLES
    print(f"Added {len(ADVERSARIAL_EXAMPLES)} adversarial examples.")
    print(f"Total golden set size: {len(all_examples)}")

    out_path = DATA_DIR / "golden_set.json"
    out_path.write_text(json.dumps(all_examples, indent=2))
    print(f"\nWrote {out_path}")
    print("\nNext: open this file, skim the 'real_pr_*' entries, fix any obviously "
          "wrong category guesses, and add more adversarial examples for failure modes "
          "you specifically want to guard against (e.g. the [code:1]-cited-7-times issue "
          "you found — turn that into an adversarial example here).")


if __name__ == "__main__":
    main()