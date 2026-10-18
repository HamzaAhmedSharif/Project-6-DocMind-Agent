"""Offline retrieval check: does the top-k context contain the evidence for each key fact?
Makes no LLM or network calls. Mirrors agents/retriever.py: same embedding model, L2 distance."""
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
CHUNK_CONFIGS = [(800, 100), (400, 50), (250, 40)]  # (chunk size in words, overlap in words)
TOP_KS = [3, 4, 5]
TOKENS_PER_WORD = 1.3  # rough English estimate, for comparing context sizes only


def main():
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
            for (q, facts), qv in zip(doc_questions, query_vecs):
                distances = ((chunk_vecs - qv) ** 2).sum(axis=1)
                context = " ".join(chunks[i] for i in np.argsort(distances)[:k])
                variants = text_variants(context)
                results = [found(kf["paper_regex"], variants) for kf in facts]
                hits += sum(results)
                total += len(results)
                full += all(results)
            rows.append({
                "chunk_size": size, "overlap": overlap, "top_k": k, "num_chunks": len(chunks),
                "fact_recall": round(hits / total, 3),
                "question_full_recall": round(full / len(doc_questions), 3),
                "approx_context_tokens": int(k * avg_words * TOKENS_PER_WORD),
            })

    print(f"{'chunk/overlap':<14}{'top_k':<7}{'fact_recall':<13}{'full_recall':<13}{'~context_tokens'}")
    for r in rows:
        print(f"{str(r['chunk_size']) + '/' + str(r['overlap']):<14}{r['top_k']:<7}"
              f"{r['fact_recall']:<13}{r['question_full_recall']:<13}{r['approx_context_tokens']}")

    os.makedirs(os.path.join("evaluation", "results"), exist_ok=True)
    out_path = os.path.join("evaluation", "results", "retrieval_grid.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": EMBEDDING_MODEL,
            "facts_used": valid_total, "facts_total": all_total,
            "rows": rows,
        }, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
