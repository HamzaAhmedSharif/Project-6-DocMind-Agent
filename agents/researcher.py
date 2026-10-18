import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retriever import Retriever
from config import TAVILY_API_KEY
from tavily import TavilyClient

retriever = Retriever()
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY and TAVILY_API_KEY != "your_key_here" else None


def research(state: dict) -> dict:
    """Researcher gathers document context, and web context if the supervisor flagged it needed."""
    doc_results = retriever.retrieve(state["query"])
    print(f"[Researcher] Retrieved {len(doc_results)} document chunks")

    web_results = []
    if state.get("needs_web_search") and tavily_client:
        print("[Researcher] Running web search via Tavily...")
        try:
            search_response = tavily_client.search(query=state["query"], max_results=3)
            web_results = [
                {"title": r["title"], "content": r["content"], "url": r["url"]}
                for r in search_response.get("results", [])
            ]
            print(f"[Researcher] Retrieved {len(web_results)} web results")
        except Exception as e:
            print(f"[Researcher] Web search failed: {e}")

    return {"doc_context": doc_results, "web_context": web_results}