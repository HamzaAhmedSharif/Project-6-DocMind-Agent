"""Offline retrieval check: does the top-k context contain the evidence for each key fact?
Makes no LLM or network calls. Mirrors agents/retriever.py: same embedding model, L2 distance.
The final settings are chosen by a rule fixed in advance (select_config), not by eye."""
import json
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer

from agents.ingest import load_pdf_text, chunk_text
from config import EMBEDDING_MODEL
from evaluation.questions import QUESTIONS
from evaluation.validate_questions import text_variants, found

PDF_PATH = os.path.join("data", "transformer_paper.pdf")
CHUNK_CONFIGS = [(800, 100), (400, 50), (250, 40), (200, 30), (150, 25)]  # (chunk size, overlap) in words
TOP_KS = [3, 4, 5, 6, 8, 10]
TOKENS_PER_WORD = 1.3      # rough English estimate, for comparing context sizes only
MAX_CONTEXT_TOKENS = 2200  # fixed in advance: leaves headroom under an 8000 tokens/min limit once web results are added
MIN_RECALL = 0.80          # fixed in advance: below this no configuration is accepted


def select_config(rows):
    """Highest fact recall within the token budget; ties go to fewer tokens, then smaller chunks."""
    eligible = [r for r in rows if r["approx_context_tokens"] <= MAX_CONTEXT_TOKENS]
    if not eligible:
        return None
    best = sorted(eligible, key=lambda r: (-r["fact_recall"], r["approx_context_tokens"], r["chunk_size"]))[0]
    return best if best["fact_recall"] >= MIN_RECALL else None


def main() -> int:
    text = load_pdf_text(PDF_PATH)
    paper_variants = text_variants(text)
    model = SentenceTransformer(EMBEDDING_MODEL)

    doc_questions = []
    all_total = valid_total = 0
    for q in QUESTIONS:
        if q["type"] != "doc":
            continue
        facts = []
        for kf in q["key_facts"]:
            all_total += 1
            if found(kf["paper_regex"], paper_variants):
                valid_total += 1
                facts.append(kf)
        if facts:
            doc_questions.append((q, facts))
    print(f"facts verified in the paper and used here: {valid_total}/{all_total}")

    query_vecs = model.encode([q["question"] for q, _ in doc_questions], convert_to_numpy=True).astype(np.float32)

    rows = []
    for size, overlap in CHUNK_CONFIGS:
        chunks = chunk_text(text, size, overlap)
        chunk_vecs = model.encode(chunks, convert_to_numpy=True).astype(np.float32)
        avg_words = float(np.mean([len(c.split()) for c in chunks]))
        for k in TOP_KS:
            hits = total = full = 0
            missed = []
            for (q, facts), qv in zip(doc_questions, query_vecs):
                distances = ((chunk_vecs - qv) ** 2).sum(axis=1)
                context = " ".join(chunks[i] for i in np.argsort(distances)[:k])
                variants = text_variants(context)
                results = [found(kf["paper_regex"], variants) for kf in facts]
                hits += sum(results)
                total += len(results)
                full += all(results)
                missed += [f"{q['id']}: {kf['fact']}" for kf, hit in zip(facts, results) if not hit]
            rows.append({
                "chunk_size": size, "overlap": overlap, "top_k": k, "num_chunks": len(chunks),
                "fact_recall": round(hits / total, 3),
                "question_full_recall": round(full / len(doc_questions), 3),
                "approx_context_tokens": int(k * avg_words * TOKENS_PER_WORD),
                "missed_facts": missed,
            })

    print(f"{'chunk/overlap':<14}{'top_k':<7}{'fact_recall':<13}{'full_recall':<13}{'~context_tokens'}")
    for r in rows:
        print(f"{str(r['chunk_size']) + '/' + str(r['overlap']):<14}{r['top_k']:<7}"
              f"{r['fact_recall']:<13}{r['question_full_recall']:<13}{r['approx_context_tokens']}")

    chosen = select_config(rows)
    os.makedirs(os.path.join("evaluation", "results"), exist_ok=True)
    out_path = os.path.join("evaluation", "results", "retrieval_grid.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": EMBEDDING_MODEL,
            "facts_used": valid_total, "facts_total": all_total,
            "selection_rule": {"max_context_tokens": MAX_CONTEXT_TOKENS, "min_fact_recall": MIN_RECALL,
                               "order": "highest fact_recall, then fewer tokens, then smaller chunk_size"},
            "selected": None if chosen is None else {k: v for k, v in chosen.items() if k != "missed_facts"},
            "rows": rows,
        }, f, indent=2)
    print("saved", out_path)

    if chosen is None:
        print("NO CONFIGURATION MEETS THE RULE (token budget and minimum recall).")
        return 1
    print(f"SELECTED: chunk_size={chosen['chunk_size']} overlap={chosen['overlap']} top_k={chosen['top_k']} "
          f"fact_recall={chosen['fact_recall']} full_recall={chosen['question_full_recall']} "
          f"approx_context_tokens={chosen['approx_context_tokens']} num_chunks={chosen['num_chunks']}")
    print("facts still missing from the selected context:")
    for m in chosen["missed_facts"]:
        print("  ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
