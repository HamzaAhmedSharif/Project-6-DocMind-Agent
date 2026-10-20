import agents.supervisor as supervisor
import agents.writer as writer

STATE = {"query": "q", "draft_answer": "draft", "doc_context": [], "web_context": [], "revision_count": 0}


def test_route_query_web(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("WEB"))
    assert supervisor.route_query({"query": "latest news"}) == {"needs_web_search": True}


def test_route_query_ignores_the_word_web_inside_a_doc_reply(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("DOC, no web needed"))
    assert supervisor.route_query({"query": "what is attention"}) == {"needs_web_search": False}


def test_route_query_is_case_and_whitespace_tolerant(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("  web\n"))
    assert supervisor.route_query({"query": "q"})["needs_web_search"] is True


def test_review_approves(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("APPROVE"))
    assert supervisor.review_answer(dict(STATE)) == {"final_answer": "draft", "review_feedback": ""}


def test_review_requests_one_revision_with_feedback(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("REVISE\nCite the source."))
    result = supervisor.review_answer(dict(STATE))
    assert result == {"revision_count": 1, "review_feedback": "Cite the source."}
    assert "final_answer" not in result


def test_review_revise_without_a_reason_uses_default_feedback(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("REVISE"))
    assert supervisor.review_answer(dict(STATE))["review_feedback"] == "Answer needs improvement."


def test_review_stops_after_one_revision(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("REVISE\nStill weak."))
    assert supervisor.review_answer({**STATE, "revision_count": 1})["final_answer"] == "draft"


def test_review_never_returns_an_empty_final_answer(monkeypatch, make_llm):
    monkeypatch.setattr(supervisor, "llm", make_llm("APPROVE"))
    result = supervisor.review_answer({**STATE, "draft_answer": ""})
    assert result["final_answer"] == "No answer could be generated."


def test_review_prompt_contains_the_evidence(monkeypatch, make_llm):
    llm = make_llm("APPROVE")
    monkeypatch.setattr(supervisor, "llm", llm)
    state = {**STATE, "doc_context": [{"source": "paper.pdf", "text": "the evidence text"}]}
    supervisor.review_answer(state)
    assert "the evidence text" in llm.prompts[0]
    assert "draft" in llm.prompts[0]


def test_format_context_lists_document_and_web_sources():
    state = {
        "doc_context": [{"source": "paper.pdf", "text": "doc text"}],
        "web_context": [{"url": "https://example.test", "title": "T", "content": "web text"}],
    }
    text = writer.format_context(state)
    assert "[Document Source: paper.pdf]" in text and "doc text" in text
    assert "[Web Source: https://example.test]" in text and "T: web text" in text


def test_format_context_says_when_nothing_was_retrieved():
    text = writer.format_context({"doc_context": [], "web_context": []})
    assert "(none retrieved)" in text and "(no web search performed)" in text


def test_write_answer_includes_reviewer_feedback_only_when_present(monkeypatch, make_llm):
    llm = make_llm("first", "second")
    monkeypatch.setattr(writer, "llm", llm)
    base = {"query": "q", "doc_context": [], "web_context": []}
    assert writer.write_answer({**base, "review_feedback": ""}) == {"draft_answer": "first"}
    assert writer.write_answer({**base, "review_feedback": "Add a citation."}) == {"draft_answer": "second"}
    assert "Add a citation." not in llm.prompts[0]
    assert "Add a citation." in llm.prompts[1]
