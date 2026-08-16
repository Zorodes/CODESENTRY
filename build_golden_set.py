"""
Builds data/golden_set.json from pr_reviews.json while filtering out stale
review comments that refer to code removed before the PR was merged.

Filtering rules (deterministic, local-only):
- If a review comment's `path` is not present in the final merged diff, discard it.
- If the file exists, compare the comment's `diff_hunk` against the final diff
  for that file using a lightweight character-based similarity (difflib).
  Normalize whitespace before comparison. Keep comments with similarity >= 0.8.

If a PR has zero remaining valid review comments after filtering, drop the
entire example (no empty `expected_findings`). Adversarial examples are
appended unchanged.

Usage: python build_golden_set.py
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict

# reuse heuristics / adversarial examples from goldenset.py
from goldenset import ADVERSARIAL_EXAMPLES, guess_category

DATA_DIR = Path(__file__).parent / "data"
PR_REVIEWS = DATA_DIR / "pr_reviews.json"
OUT_PATH = DATA_DIR / "golden_set.json"
SIMILARITY_THRESHOLD = 0.8
MAX_REAL_EXAMPLES = 30

FILE_DIFF_SPLIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def normalize_text(s: str) -> str:
    # collapse all whitespace to single space and strip
    return re.sub(r"\s+", " ", s or "").strip()


def split_diff_by_file(full_diff: str) -> Dict[str, str]:
    """Split a unified diff into a mapping of file path -> file diff text.

    Uses the `b/` side path as the canonical key (matches `review_comment['path']`).
    """
    parts = {}
    if not full_diff:
        return parts

    # find all diff headers and their spans
    matches = list(FILE_DIFF_SPLIT_RE.finditer(full_diff))
    if not matches:
        return parts

    for i, m in enumerate(matches):
        a_path = m.group(1)
        b_path = m.group(2)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_diff)
        section = full_diff[start:end]
        # use the b/ path (file as in repo after merge)
        parts[b_path] = section
    return parts


def comment_matches_file(hunk: str, file_diff: str, threshold: float) -> bool:
    """Return True if normalized `hunk` is similar enough to `file_diff`."""
    if not hunk or not file_diff:
        return False
    a = normalize_text(hunk)
    b = normalize_text(file_diff)
    # quick check to avoid heavy ratio calls on tiny strings
    if not a or not b:
        return False
    # SequenceMatcher gives a deterministic char-based similarity score
    score = SequenceMatcher(None, a, b).ratio()
    return score >= threshold


def build_real_examples_filtered() -> list:
    if not PR_REVIEWS.exists():
        raise FileNotFoundError(f"{PR_REVIEWS} not found — run ingest.py first to create it")

    prs = json.loads(PR_REVIEWS.read_text())
    examples = []

    for pr in prs:
        diff = pr.get("diff", "")
        if not diff:
            continue

        file_diffs = split_diff_by_file(diff)
        if not file_diffs:
            continue

        expected_findings = []

        # process inline review comments (they usually include path + diff_hunk)
        for c in pr.get("review_comments", []):
            body = c.get("body", "").strip()
            if len(body) < 10:
                continue
            path = c.get("path")
            hunk = c.get("diff_hunk")
            # rule 1: file must exist in final diff
            if not path or path not in file_diffs:
                continue
            # rule 2: diff_hunk must be similar to that file's final diff
            if not hunk:
                # no hunk to check — conservatively drop (cannot verify against final diff)
                continue
            if not comment_matches_file(hunk, file_diffs[path], SIMILARITY_THRESHOLD):
                continue

            expected_findings.append({
                "category": guess_category(body),
                "should_flag": True,
                "description": body[:300],
            })

        # process top-level reviews (rarely include a path/hunk) — only keep if path+hunk present
        for r in pr.get("reviews", []):
            body = r.get("body", "").strip()
            if len(body) < 10 or body.lower().strip() in ("lgtm", "looks good", "approved"):
                continue
            # some review records may include a path/hunk in an expanded form; try to use them
            path = r.get("path")
            hunk = r.get("diff_hunk")
            if not path or path not in file_diffs or not hunk:
                continue
            if not comment_matches_file(hunk, file_diffs[path], SIMILARITY_THRESHOLD):
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


def main():
    print("Building golden set with stale-comment filtering...")
    real = build_real_examples_filtered()
    print(f"Kept {len(real)} real PR examples after filtering (max {MAX_REAL_EXAMPLES}).")
    all_examples = real + ADVERSARIAL_EXAMPLES
    print(f"Appending {len(ADVERSARIAL_EXAMPLES)} adversarial examples; total {len(all_examples)}.")
    OUT_PATH.write_text(json.dumps(all_examples, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
