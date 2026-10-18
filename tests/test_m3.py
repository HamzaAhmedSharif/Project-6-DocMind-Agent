import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import run_query
from agents.llm import STATS, reset_stats

if __name__ == "__main__":
    query = input("Ask a question (multi-agent DocMind): ")
    reset_stats()
    result = run_query(query)

    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])
    print("\n--- DOCUMENT SOURCES ---")
    print(list(set(r["source"] for r in result.get("doc_context", []))))
    print("\n--- WEB SOURCES ---")
    print([r["url"] for r in result.get("web_context", [])])
    print("\n--- LLM STATS ---")
    print(STATS)
