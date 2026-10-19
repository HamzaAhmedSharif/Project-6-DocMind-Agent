"""Paraphrase review layer. Strict regex scoring counts a correct paraphrase as a miss; this module lets a
reviewer overturn a strict miss, but only with a quote that is verified against the saved answer,
so no quote can be invented."""
import json
import os

from evaluation.questions import QUESTIONS
from evaluation.scoring import clean

ADJUDICATION_PATH = os.path.join("evaluation", "adjudication.json")
QMAP = {q["id"]: q for q in QUESTIONS}


def load_entries(path=ADJUDICATION_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]


def apply(results, scores, entries):
    """Return (overturned strict misses per system, problems). Invalid or unverifiable entries are not counted."""
    extra, problems, seen = {}, [], set()
    for e in entries:
        qid, system, idx, quote = e["id"], e["system"], e["fact"], e["quote"]
        key = (qid, system, idx)
        entry = results.get(qid, {}).get(system)
        sc = scores.get(qid, {}).get(system)
        if key in seen:
            problems.append(f"{qid}/{system}/fact {idx}: duplicate entry")
        elif entry is None or "error" in entry or sc is None or QMAP[qid]["type"] != "doc":
            problems.append(f"{qid}/{system}/fact {idx}: no complete answer to review")
        elif sc["hits"][idx]:
            problems.append(f"{qid}/{system}/fact {idx}: already a strict hit")
        elif clean(quote).lower() not in clean(entry["answer"]).lower():
            problems.append(f"{qid}/{system}/fact {idx}: quote not found in the saved answer")
        else:
            extra[system] = extra.get(system, 0) + 1
        seen.add(key)
    return extra, problems
