"""Checks the evaluation set against the paper's own text. Makes no LLM or network calls."""
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ingest import load_pdf_text
from evaluation.questions import QUESTIONS

PDF_PATH = os.path.join("data", "transformer_paper.pdf")
EXPECTED_COUNTS = {"doc": 12, "web": 4, "absent": 4}


def text_variants(text: str) -> list:
    """Whitespace-collapsed text, plus a variant with hyphenated line breaks joined."""
    collapsed = re.sub(r"\s+", " ", text)
    joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", collapsed)
    return [collapsed, joined]


def found(pattern: str, variants: list) -> bool:
    return any(re.search(pattern, v, re.IGNORECASE) for v in variants)


def diagnose(fact: str, variants: list) -> list:
    """For a fact that was not found, show how its key words actually appear in the extracted text."""
    lines = []
    collapsed = variants[0]
    for word in re.findall(r"[A-Za-z]{5,}", fact)[:3]:
        match = re.search(re.escape(word), collapsed, re.IGNORECASE)
        if match:
            start, end = max(0, match.start() - 60), min(len(collapsed), match.end() + 60)
            lines.append(f"    '{word}' -> ...{collapsed[start:end]}...")
        else:
            lines.append(f"    '{word}' -> not found in extracted text")
    return lines


def main() -> int:
    paper_text = load_pdf_text(PDF_PATH)
    variants = text_variants(paper_text)
    print(f"paper characters extracted: {len(paper_text)}")

    counts = {"doc": 0, "web": 0, "absent": 0}
    seen_ids = set()
    problems = []
    facts_total = facts_found = 0

    for q in QUESTIONS:
        counts[q["type"]] += 1
        if q["id"] in seen_ids:
            problems.append(f"{q['id']}: duplicate id")
        seen_ids.add(q["id"])

        if q["type"] == "doc":
            for kf in q["key_facts"]:
                facts_total += 1
                for field in ("answer_regex", "paper_regex"):
                    try:
                        re.compile(kf[field])
                    except re.error as e:
                        problems.append(f"{q['id']}: invalid {field} for '{kf['fact']}': {e}")
                if found(kf["paper_regex"], variants):
                    facts_found += 1
                else:
                    problems.append(f"{q['id']}: fact NOT found in paper: {kf['fact']}")
                    problems.extend(diagnose(kf["fact"], variants))
        elif q["type"] == "absent":
            for pattern in q["absent_regex"]:
                if found(pattern, variants):
                    problems.append(f"{q['id']}: pattern unexpectedly present in paper: {pattern}")
        elif q["type"] == "web" and q.get("expected_route") != "WEB":
            problems.append(f"{q['id']}: web question must have expected_route 'WEB'")

    if counts != EXPECTED_COUNTS:
        problems.append(f"question counts {counts} != expected {EXPECTED_COUNTS}")

    print(f"question counts: {counts}")
    print(f"fact checks: {facts_found}/{facts_total} found in the paper")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" ", p)
        return 1
    print("OK: every key fact is present in the paper and every absent topic is genuinely absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
