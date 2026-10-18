import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm import get_llm
from agents.writer import format_context

llm = get_llm("supervisor")


def route_query(state: dict) -> dict:
    """Supervisor decides if this query needs live web search beyond the local document."""
    prompt = f"""You are a routing supervisor. A user asked a question about a document (a research paper).
Decide if answering this question well requires CURRENT web information (recent events, latest versions, things published after the document), or if it can be answered from the document alone.

Question: {state['query']}

Reply with exactly one word: WEB or DOC"""

    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    needs_web = decision.startswith("WEB")

    print(f"[Supervisor] Routing decision: {'web search + document' if needs_web else 'document only'}")

    return {"needs_web_search": needs_web}


def review_answer(state: dict) -> dict:
    """Supervisor reviews the drafted answer against the same evidence the writer was given."""
    prompt = f"""You are a strict quality reviewer. Check the answer below against the context it was written from.
It must (1) actually answer the question and (2) make only claims supported by the context.

Question: {state['query']}

{format_context(state)}

Answer to review:
{state['draft_answer']}

Reply with exactly one word: APPROVE or REVISE. If REVISE, add a one-sentence reason on a new line."""

    response = llm.invoke(prompt)
    result = response.content.strip()
    approved = result.upper().startswith("APPROVE")

    feedback = ""
    if not approved:
        lines = result.split("\n", 1)
        feedback = lines[1] if len(lines) > 1 else "Answer needs improvement."

    print(f"[Supervisor] Review: {'APPROVED' if approved else 'REVISE - ' + feedback}")

    if approved or state.get("revision_count", 0) >= 1:
        return {
            "final_answer": state["draft_answer"] or "No answer could be generated.",
            "review_feedback": feedback,
        }
    else:
        return {"revision_count": state.get("revision_count", 0) + 1, "review_feedback": feedback}
