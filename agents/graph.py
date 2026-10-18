import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.supervisor import route_query, review_answer
from agents.researcher import research
from agents.writer import write_answer


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor_router", route_query)
    workflow.add_node("researcher", research)
    workflow.add_node("writer", write_answer)
    workflow.add_node("supervisor_review", review_answer)

    workflow.set_entry_point("supervisor_router")
    workflow.add_edge("supervisor_router", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "supervisor_review")

    def route_after_review(state):
        return "end" if state.get("final_answer") else "revise"

    workflow.add_conditional_edges(
        "supervisor_review",
        route_after_review,
        {"end": END, "revise": "writer"}
    )

    return workflow.compile()


def run_query(query: str) -> dict:
    graph = build_graph()
    initial_state = {
        "query": query,
        "needs_web_search": False,
        "doc_context": [],
        "web_context": [],
        "draft_answer": "",
        "final_answer": "",
        "revision_count": 0,
        "review_feedback": ""
    }
    return graph.invoke(initial_state)