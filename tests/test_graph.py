import importlib
import sys
import types

import pytest

import agents.supervisor as supervisor
import agents.writer as writer


@pytest.fixture
def graph_module(monkeypatch):
    """agents.graph imported with a fake researcher: no index, embedding model or Tavily is needed."""
    researcher_calls = []

    def research(state):
        researcher_calls.append(state["needs_web_search"])
        web = [{"title": "T", "content": "web text", "url": "https://example.test"}] if state["needs_web_search"] else []
        return {"doc_context": [{"source": "paper.pdf", "text": "doc text"}], "web_context": web}

    fake_researcher = types.ModuleType("agents.researcher")
    fake_researcher.research = research
    monkeypatch.setitem(sys.modules, "agents.researcher", fake_researcher)
    monkeypatch.delitem(sys.modules, "agents.graph", raising=False)
    module = importlib.import_module("agents.graph")
    module.researcher_calls = researcher_calls
    yield module
    sys.modules.pop("agents.graph", None)


def run_graph(graph_module, monkeypatch, make_llm, router, drafts, reviews):
    supervisor_llm = make_llm(router, *reviews)  # the supervisor model routes first, then reviews
    writer_llm = make_llm(*drafts)
    monkeypatch.setattr(supervisor, "llm", supervisor_llm)
    monkeypatch.setattr(writer, "llm", writer_llm)
    result = graph_module.run_query("What is attention?")
    return result, supervisor_llm, writer_llm


def test_document_question_is_approved_on_the_first_pass(graph_module, monkeypatch, make_llm):
    result, sup, wri = run_graph(graph_module, monkeypatch, make_llm, "DOC", ["draft one"], ["APPROVE"])
    assert result["final_answer"] == "draft one"
    assert result["needs_web_search"] is False
    assert result["revision_count"] == 0
    assert graph_module.researcher_calls == [False]
    assert len(sup.prompts) == 2 and len(wri.prompts) == 1  # router + reviewer, one draft


def test_one_revision_then_approval(graph_module, monkeypatch, make_llm):
    result, sup, wri = run_graph(
        graph_module, monkeypatch, make_llm, "DOC", ["draft one", "draft two"], ["REVISE\nAdd a citation.", "APPROVE"]
    )
    assert result["final_answer"] == "draft two"
    assert result["revision_count"] == 1
    assert len(wri.prompts) == 2
    assert "Add a citation." in wri.prompts[1]


def test_revision_is_capped_at_one(graph_module, monkeypatch, make_llm):
    result, sup, wri = run_graph(
        graph_module, monkeypatch, make_llm, "DOC", ["draft one", "draft two"], ["REVISE\nA.", "REVISE\nB."]
    )
    assert result["final_answer"] == "draft two"
    assert len(wri.prompts) == 2 and len(sup.prompts) == 3  # router + two reviews


def test_web_question_passes_web_context_to_writer_and_reviewer(graph_module, monkeypatch, make_llm):
    result, sup, wri = run_graph(graph_module, monkeypatch, make_llm, "WEB", ["web draft"], ["APPROVE"])
    assert result["needs_web_search"] is True
    assert graph_module.researcher_calls == [True]
    assert "[Web Source: https://example.test]" in wri.prompts[0]
    assert "[Web Source: https://example.test]" in sup.prompts[1]


def test_empty_draft_cannot_loop_forever(graph_module, monkeypatch, make_llm):
    result, _, _ = run_graph(graph_module, monkeypatch, make_llm, "DOC", [""], ["APPROVE"])
    assert result["final_answer"] == "No answer could be generated."
