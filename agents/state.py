from typing import TypedDict, List


class AgentState(TypedDict):
    query: str
    needs_web_search: bool
    doc_context: List[dict]
    web_context: List[dict]
    draft_answer: str
    final_answer: str
    revision_count: int
    review_feedback: str