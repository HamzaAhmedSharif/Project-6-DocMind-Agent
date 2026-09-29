import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retriever import Retriever
from agents.rag_chain import answer_question

if __name__ == "__main__":
    retriever = Retriever()
    query = input("Ask a question about your document(s): ")
    result = answer_question(query, retriever)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    print(result["sources"])