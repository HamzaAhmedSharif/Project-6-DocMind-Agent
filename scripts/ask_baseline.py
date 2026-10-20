import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retriever import Retriever
from agents.rag_chain import answer_question
from agents.llm import STATS, reset_stats

if __name__ == "__main__":
    retriever = Retriever()
    query = input("Ask a question about your document(s): ")
    reset_stats()
    result = answer_question(query, retriever)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    print(result["sources"])
    print("\n--- LLM STATS ---")
    print(STATS)
