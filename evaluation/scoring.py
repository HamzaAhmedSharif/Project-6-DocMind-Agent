"""Pure scoring functions for the comparison. No network, no LLM. Raw outputs are stored separately,
so scoring can be corrected and rerun without spending tokens."""
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

ABSTAIN_PATTERNS = [
    r"(does\s*not|doesn't|did\s*not|didn't|do\s*not|don't)\s*(contain|mention|provide|include|report|specify|state|cover|discuss|address|give|have|say|offer|present)",
    r"\bnot\s*(mentioned|provided|reported|specified|stated|covered|discussed|addressed|included|found|available|present|given)\b",
    r"\bno\s*(information|mention|data|evidence|details|reference|record)\b",
    r"\b(cannot|can't|can\s*not|unable\s*to)\s*(find|answer|determine|locate|confirm)",
    r"\b(isn't|is\s*not|wasn't|was\s*not|aren't|are\s*not)\s*(mentioned|provided|reported|specified|stated|covered|discussed|included|available|present|given)",
    r"\b(not|n't)\s*(in|within)\s*the\s*(provided\s*|given\s*)?(context|document|paper|text|excerpt)",
]


def clean(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    return re.sub(r"\s+", " ", text).strip()


def score_key_facts(answer: str, key_facts: list) -> dict:
    """Per-fact hits for an answer to an answerable question."""
    cleaned = clean(answer)
    hits = [bool(re.search(kf["answer_regex"], cleaned, re.IGNORECASE)) for kf in key_facts]
    return {"hits": hits, "found": sum(hits), "total": len(hits), "all_found": all(hits)}


def is_abstention(answer: str) -> bool:
    """True if the answer says the information is not available or not in the document."""
    cleaned = clean(answer)
    return any(re.search(p, cleaned, re.IGNORECASE) for p in ABSTAIN_PATTERNS)
