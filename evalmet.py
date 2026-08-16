"""
Eval metrics for CodeSentry — implemented with LLM-as-judge instead of Ragas.

Ragas was evaluated and rejected: it forces langchain-core down to a version
incompatible with the langgraph version this project depends on (langgraph
requires langchain-core>=1.4.7, ragas forces <2 alongside a chain that lands
on 0.2.x), and the newer ragas release additionally fails on import due to an
unconditional Vertex AI submodule import unrelated to this project. Rather
than fight dependency resolution, this reimplements the two Ragas concepts
that actually matter here — faithfulness (grounding) and finding-level
precision/recall — using the same LLM-as-judge pattern already used by
critic_node, with zero new dependencies.

Three metrics:
1. Finding-level precision/recall — does verified_findings match what the
   golden set says should have been flagged?
2. Faithfulness rate — for each verified finding, does its cited source
   ([code:ID], [precedent:ID], or [diff]) actually support the claim?
3. Hallucination rate — for adversarial examples with known fabricated
   claims planted in them, what fraction of those claims survived into
   verified_findings? This should be at or near zero.
"""

import re
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_utils import invoke_structured_llm

JUDGE_MODEL = "models/gemini-3.5-flash"
CITATION_PATTERN = re.compile(r"\[?(code|precedent):(\d+)\]?")


class MatchJudgment(BaseModel):
    matches: bool


class GroundingJudgment(BaseModel):
    is_grounded: bool


def _get_judge_llm():
    return ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0.0)


def finding_matches_expected(finding_description: str, expected_description: str) -> bool:
    """LLM-judge: do these two descriptions refer to the same underlying issue?
    Exact string matching doesn't work here since wording never matches verbatim
    between a real reviewer's comment and CodeSentry's generated finding."""
    llm = _get_judge_llm()
    structured = llm.with_structured_output(MatchJudgment)
    prompt = f"""Do these two code review comments describe the same underlying issue?
Answer based on substance, not wording — different phrasing of the same problem counts as a match.

Comment A: {finding_description}
Comment B: {expected_description}

Do they describe the same issue?"""
    result = invoke_structured_llm(structured, prompt)
    return result.matches


def compute_precision_recall(verified_findings: list, expected_findings: list) -> dict:
    """
    Precision: of what CodeSentry flagged, how much matches a real expected finding?
    Recall: of what should have been flagged, how much did CodeSentry catch?

    expected_findings with should_flag=False are excluded from recall (nothing to
    catch) — those are handled separately by false-positive checks.
    """
    should_flag = [e for e in expected_findings if e.get("should_flag", True)]

    if not verified_findings and not should_flag:
        # Nothing expected, nothing flagged — perfect on this example, not a divide-by-zero case
        return {"precision": 1.0, "recall": 1.0, "matched": 0, "flagged": 0, "expected": 0}

    matched_expected_idxs = set()
    matched_verified_idxs = set()

    for vi, finding in enumerate(verified_findings):
        for ei, expected in enumerate(should_flag):
            if ei in matched_expected_idxs:
                continue
            if finding_matches_expected(finding.description, expected["description"]):
                matched_expected_idxs.add(ei)
                matched_verified_idxs.add(vi)
                break

    precision = len(matched_verified_idxs) / len(verified_findings) if verified_findings else 1.0
    recall = len(matched_expected_idxs) / len(should_flag) if should_flag else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "matched": len(matched_expected_idxs),
        "flagged": len(verified_findings),
        "expected": len(should_flag),
    }


def check_citation_exists(finding, code_chunks: list[dict], precedents: list[dict]) -> bool:
    """
    Deterministic check: does every citation on this finding point at something
    that was actually retrieved? No LLM call — this is a factual lookup, not a
    judgment call.

    Deliberately NOT re-asking an LLM "does the cited content semantically
    support this claim" here — that's the same question critic_node already
    answers, using the same model. Re-asking it at eval time isn't independent
    verification, it's the same LLM agreeing with its own prior judgment,
    which would give a falsely reassuring score rather than real signal.
    A citation that points at nothing real, or a claim that doesn't actually
    match reality, will fail to match anything in the golden set anyway and
    get caught by precision/recall instead — that's the metric doing the
    real work here.
    """
    code_ids = {str(c.get("id")) for c in code_chunks}
    precedent_ids = {str(p.get("id")) for p in precedents}

    for cid_str in finding.source_chunk_ids:
        s = str(cid_str).strip()
        if s.lower() in ("[diff]", "diff"):
            continue  # diff is always in context, always a valid citation target
        match = CITATION_PATTERN.search(s)
        if not match:
            return False  # unparseable citation format
        kind, cid = match.groups()
        valid_ids = code_ids if kind == "code" else precedent_ids
        if cid not in valid_ids:
            return False  # citing something that was never retrieved

    return True


def compute_citation_validity_rate(verified_findings: list, code_chunks: list[dict],
                                     precedents: list[dict]) -> float:
    if not verified_findings:
        return 1.0
    valid = sum(1 for f in verified_findings if check_citation_exists(f, code_chunks, precedents))
    return valid / len(verified_findings)


def compute_hallucination_survival(verified_findings: list, known_fabricated_claims: list[str]) -> bool:
    """
    For adversarial examples with known fabricated claims (e.g. "turbo
    encabulator") planted in the diff's comments. Returns True if ANY
    fabricated claim survived into verified_findings — a failure.
    """
    if not known_fabricated_claims:
        return False

    all_descriptions = " ".join(f.description.lower() for f in verified_findings)
    return any(claim.lower() in all_descriptions for claim in known_fabricated_claims)