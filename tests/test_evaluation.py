import re

import pytest

import evaluation.report as report_mod
from evaluation.adjudication import apply as apply_review
from evaluation.questions import QUESTIONS
from evaluation.report import build_markdown, score_run
from evaluation.scoring import clean, is_abstention, score_key_facts

QMAP = {q["id"]: q for q in QUESTIONS}
COSTS = {"calls": 1, "prompt_tokens": 100, "completion_tokens": 10,
         "api_seconds": 1.0, "throttle_seconds": 0.0, "wall_seconds": 1.0}


def entry(answer, route=None):
    data = {"answer": answer, **COSTS}
    if route is not None:
        data.update({"route": route, "revisions": 0, "web_urls": []})
    return data


def test_question_set_has_the_expected_shape():
    counts = {}
    for q in QUESTIONS:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    assert counts == {"doc": 12, "web": 4, "absent": 4}
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    assert all(q["expected_route"] == "WEB" for q in QUESTIONS if q["type"] == "web")


def test_every_pattern_compiles():
    for q in QUESTIONS:
        for kf in q.get("key_facts", []):
            re.compile(kf["answer_regex"])
            re.compile(kf["paper_regex"])
        for pattern in q.get("absent_regex", []):
            re.compile(pattern)


def test_clean_normalizes_typography():
    assert clean("Scaled Dot\u2011Product") == "Scaled Dot-Product"
    assert clean("warm\u2011up of 4\u202f000 steps") == "warm-up of 4000 steps"
    assert clean("it\u2019s") == "it's"


def test_thousands_separators_are_not_applied_to_unrelated_numbers():
    assert clean("28.4, 41.8 and 512 2048") == "28.4, 41.8 and 512 2048"


def test_score_key_facts_counts_matches_after_normalization():
    answer = "Scaled Dot\u2011Product Attention applies a softmax after dividing by sqrt(d_k)."
    result = score_key_facts(answer, QMAP["d02"]["key_facts"])
    assert result["found"] == 3 and result["all_found"] is True


def test_score_key_facts_reports_misses():
    result = score_key_facts("It is a kind of pooling.", QMAP["d02"]["key_facts"])
    assert result["found"] == 0 and result["hits"] == [False, False, False]


@pytest.mark.parametrize("text,expected", [
    ("The paper does not mention ImageNet.", True),
    ("The paper doesn\u2019t mention ImageNet.", True),
    ("There is no information about this in the provided context.", True),
    ("I cannot answer the question from the given sources.", True),
    ("The Transformer reached 28.4 BLEU.", False),
])
def test_is_abstention(text, expected):
    assert is_abstention(text) is expected


def test_review_overturns_a_strict_miss_only_with_a_verifiable_quote():
    answer = "The decoder blocks each position from looking ahead to tokens that have not yet been generated."
    results = {"d12": {"type": "doc", "question": "q", "baseline": entry(answer)}}
    scores = score_run(results)
    assert scores["d12"]["baseline"]["hits"][0] is False
    good = [{"id": "d12", "system": "baseline", "fact": 0, "quote": "looking ahead to tokens that have not yet been generated"}]
    assert apply_review(results, scores, good) == ({"baseline": 1}, [])
    invented = [{"id": "d12", "system": "baseline", "fact": 0, "quote": "attends only to positions to the left"}]
    extra, problems = apply_review(results, scores, invented)
    assert extra == {} and "quote not found" in problems[0]


def test_review_rejects_strict_hits_and_duplicates():
    answer = "Masking prevents attention to future positions, so predictions use earlier positions only."
    results = {"d12": {"type": "doc", "question": "q", "baseline": entry(answer)}}
    scores = score_run(results)
    assert scores["d12"]["baseline"]["hits"][0] is True
    entries = [{"id": "d12", "system": "baseline", "fact": 0, "quote": "future positions"}] * 2
    extra, problems = apply_review(results, scores, entries)
    assert extra == {}
    assert any("already a strict hit" in p for p in problems)
    assert any("duplicate" in p for p in problems)


def test_url_in_answer_is_checked_in_the_text_not_the_state():
    multi = {**entry("A table of developments.", route="WEB"), "web_urls": ["https://example.test"]}
    item = score_run({"w01": {"type": "web", "question": "q", "multi": multi}})["w01"]["multi"]
    assert item["web_results_retrieved"] is True
    assert item["url_in_answer"] is False


def test_report_markdown_contains_the_headline_rows(monkeypatch):
    monkeypatch.setattr(report_mod, "load_entries", lambda: [])
    good = "Scaled Dot\u2011Product Attention applies a softmax after dividing by sqrt(d_k)."
    results = {"d02": {"type": "doc", "question": "q", "baseline": entry(good),
                       "single_writer": entry(good), "multi": entry(good, route="DOC")}}
    run = {"started": "2026-01-01 00:00:00",
           "fingerprint": {"settings": {"model": "m", "provider": "p", "extra_body": {}, "chunk_size": 150,
                                        "chunk_overlap": 25, "top_k": 10, "embedding_model": "e"}},
           "results": results}
    markdown = build_markdown(run, results, score_run(results))
    assert "100.0% (3/3)" in markdown
    assert "Web questions: web results retrieved" in markdown
    assert "Web questions: answer text contains a URL" in markdown
    assert "coverage after paraphrase review" in markdown
    assert "has_web_urls" not in markdown
