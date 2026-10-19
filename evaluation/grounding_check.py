"""Offline check of whether stated key facts are supported by what was retrieved. No LLM or network calls.
For every answerable question and system, a key fact the answer states (strict scoring) is 'supported' if its
evidence (paper_regex) is present in the retrieved context for that question. A stated fact whose evidence is
absent from the context may have come from the model's own knowledge (approximate: the check is regex based)."""
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retriever import Retriever
from evaluation.questions import QUESTIONS
from evaluation.scoring import score_key_facts
from evaluation.validate_questions import text_variants, found

RUN_PATH = os.path.join("evaluation", "results", "run_main.json")
OUT_PATH = os.path.join("evaluation", "results", "grounding_check.json")
SYSTEMS = ("baseline", "single_writer", "multi")


def main():
    with open(RUN_PATH, encoding="utf-8") as f:
        results = json.load(f)["results"]
    retriever = Retriever()

    summary = {s: {"facts_stated": 0, "supported": 0, "unsupported": []} for s in SYSTEMS}
    context_missing = []
    for q in QUESTIONS:
        if q["type"] != "doc":
            continue
        context = " ".join(r["text"] for r in retriever.retrieve(q["question"]))
        variants = text_variants(context)
        in_context = [found(kf["paper_regex"], variants) for kf in q["key_facts"]]
        context_missing += [f"{q['id']}: {kf['fact']}" for kf, ok in zip(q["key_facts"], in_context) if not ok]
        for system in SYSTEMS:
            entry = results.get(q["id"], {}).get(system)
            if not entry or "error" in entry:
                continue
            hits = score_key_facts(entry["answer"], q["key_facts"])["hits"]
            for kf, hit, ok in zip(q["key_facts"], hits, in_context):
                if not hit:
                    continue
                summary[system]["facts_stated"] += 1
                if ok:
                    summary[system]["supported"] += 1
                else:
                    summary[system]["unsupported"].append(f"{q['id']}: {kf['fact']}")

    print(f"key facts whose evidence is absent from the retrieved context: {len(context_missing)}")
    for m in context_missing:
        print("  ", m)
    print()
    print(f"{'system':<15}{'facts stated (strict)':<24}{'evidence in context':<22}{'stated without evidence in context'}")
    for s in SYSTEMS:
        d = summary[s]
        print(f"{s:<15}{d['facts_stated']:<24}{d['supported']:<22}{len(d['unsupported'])}")
        for u in d["unsupported"]:
            print("    ", u)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"facts_absent_from_context": context_missing, "per_system": summary}, f, indent=2)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
