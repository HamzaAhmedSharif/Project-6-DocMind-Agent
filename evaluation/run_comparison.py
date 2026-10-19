"""Runs the single-agent baseline and the multi-agent pipeline on every evaluation question.

Raw outputs are saved after every call, so a quota stop can be resumed by running the same command again.
Resuming is refused if the pipeline code or settings changed since the run started (results must not mix).

Usage: python evaluation/run_comparison.py [--out PATH] [--only d01,d02]
"""
import argparse
import hashlib
import json
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.llm import STATS, reset_stats
from agents.retriever import Retriever
from agents.rag_chain import answer_question
from agents.graph import run_query
from evaluation.questions import QUESTIONS

DEFAULT_OUT = os.path.join("evaluation", "results", "run_main.json")
FROZEN_FILES = [os.path.join("agents", name) for name in
                ("supervisor.py", "writer.py", "rag_chain.py", "researcher.py", "graph.py")]


def fingerprint() -> dict:
    digest = hashlib.sha256()
    for path in FROZEN_FILES:
        with open(path, "rb") as f:
            digest.update(f.read().replace(b"\r\n", b"\n"))
    settings = {
        "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL, "extra_body": config.LLM_EXTRA_BODY,
        "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K_RESULTS,
        "embedding_model": config.EMBEDDING_MODEL,
    }
    return {"code_sha256": digest.hexdigest(), "settings": settings}


def snapshot(started: float) -> dict:
    return {
        "wall_seconds": round(time.time() - started, 2),
        "calls": STATS["calls"],
        "api_seconds": round(STATS["api_seconds"], 2),
        "throttle_seconds": round(STATS["throttle_seconds"], 2),
        "prompt_tokens": STATS["prompt_tokens"],
        "completion_tokens": STATS["completion_tokens"],
    }


def run_baseline(question, retriever):
    reset_stats()
    started = time.time()
    result = answer_question(question, retriever)
    return {"answer": result["answer"], "doc_sources": sorted(result["sources"]), **snapshot(started)}


def run_multi(question):
    reset_stats()
    started = time.time()
    state = run_query(question)
    return {
        "answer": state["final_answer"],
        "route": "WEB" if state.get("needs_web_search") else "DOC",
        "revisions": state.get("revision_count", 0),
        "review_feedback": state.get("review_feedback", ""),
        "doc_sources": sorted({r["source"] for r in state.get("doc_context", [])}),
        "web_urls": [r["url"] for r in state.get("web_context", [])],
        **snapshot(started),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--only", default="", help="comma-separated question ids (for debugging)")
    args = parser.parse_args()

    current = fingerprint()
    print("code fingerprint:", current["code_sha256"][:12], "settings:", current["settings"])
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            run = json.load(f)
        if run["fingerprint"] != current:
            print("REFUSING TO RESUME: prompts, pipeline code or settings changed since this run started.")
            print("Start a new run with a different --out file.")
            return 3
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        run = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "fingerprint": current, "results": {}}

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    retriever = Retriever()

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, ensure_ascii=False)

    for q in QUESTIONS:
        if only and q["id"] not in only:
            continue
        entry = run["results"].setdefault(q["id"], {"type": q["type"], "question": q["question"]})
        for system in ("baseline", "multi"):
            if system in entry and "error" not in entry[system]:
                continue  # already complete
            print(f"\n=== {q['id']} [{q['type']}] {system} ===")
            try:
                if system == "baseline":
                    entry[system] = run_baseline(q["question"], retriever)
                else:
                    entry[system] = run_multi(q["question"])
            except RuntimeError as e:
                if str(e).startswith("Daily quota exhausted"):
                    save()
                    print(f"STOPPED: {e}")
                    print(f"Progress saved to {args.out}. Run the same command again after the quota resets.")
                    return 2
                entry[system] = {"error": str(e)[:500]}
            except Exception as e:  # keep going; the error is recorded and retried on the next run
                entry[system] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
            save()

    done = sum(1 for e in run["results"].values() for s in ("baseline", "multi") if s in e and "error" not in e[s])
    errors = sum(1 for e in run["results"].values() for s in ("baseline", "multi") if s in e and "error" in e[s])
    print(f"\nFinished pass: {done} results complete, {errors} with errors (rerun the same command to retry errors).")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
