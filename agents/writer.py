import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm import get_llm

llm = get_llm("writer")


def format_context(state: dict) -> str:
    """Render document and web context exactly as the writer and the reviewer see it."""
    doc_context = "\n\n".join(
        [f"[Document Source: {r['source']}]\n{r['text']}" for r in state.get("doc_context", [])]
    )
    web_context = "\n\n".join(
        [f"[Web Source: {r['url']}]\n{r['title']}: {r['content']}" for r in state.get("web_context", [])]
    )
    return (
        f"Document Context:\n{doc_context if doc_context else '(none retrieved)'}\n\n"
        f"Web Context:\n{web_context if web_context else '(no web search performed)'}"
    )


def write_answer(state: dict) -> dict:
    """Writer drafts an answer using document context and web context (if any)."""
    feedback_note = ""
    if state.get("review_feedback"):
        feedback_note = f"\n\nA reviewer flagged your previous draft with this feedback, address it: {state['review_feedback']}"

    prompt = f"""Answer the question using the context below. Cite which source(s) you used (document or web).

{format_context(state)}

Question: {state['query']}{feedback_note}

Answer:"""

    response = llm.invoke(prompt)
    print("[Writer] Draft answer generated")

    return {"draft_answer": response.content}
