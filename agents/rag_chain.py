import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm import get_llm
from agents.retriever import Retriever

llm = get_llm("writer")


def answer_question(query: str, retriever: Retriever) -> dict:
    results = retriever.retrieve(query)

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    context = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for r in results])

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so clearly. Cite which source(s) you used.

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": list(set(r["source"] for r in results))
    }