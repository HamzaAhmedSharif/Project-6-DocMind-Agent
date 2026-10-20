import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

import agents.llm as llm_mod
from agents.llm import ThrottledLLM, TokenBucket, parse_retry_after


@pytest.fixture(autouse=True)
def clean_pacing_state():
    llm_mod._last_call_time.clear()
    llm_mod._buckets.clear()
    llm_mod.reset_stats()
    yield


@pytest.fixture
def clock(monkeypatch, fake_time):
    monkeypatch.setattr(llm_mod, "time", fake_time)
    return fake_time


class FakeChat:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ai(text, prompt_tokens=100, completion_tokens=20):
    return AIMessage(
        content=text,
        response_metadata={"token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    )


def status_error(cls, message, code):
    response = httpx.Response(code, request=httpx.Request("POST", "http://example.test"))
    return cls(message, response=response, body=None)


def throttled(chat, bucket=None, min_interval=0.0):
    t = ThrottledLLM("test-model", 0.0)
    t.min_interval = min_interval
    t.bucket = bucket
    t.llm = chat
    return t


def test_parse_retry_after():
    assert parse_retry_after("Please try again in 1m26.4s. Visit https://x") == pytest.approx(87.4)
    assert parse_retry_after("try again in 250ms") == pytest.approx(1.25)
    assert parse_retry_after("try again in 2s") == pytest.approx(3.0)
    assert parse_retry_after("no hint here") == 60.0
    assert parse_retry_after("no hint here", default=5.0) == 5.0


def test_token_bucket_allows_a_full_budget_then_waits_for_refill(clock):
    bucket = TokenBucket(7200)  # refills 120 tokens per second
    assert bucket.reserve(5000) == 0.0
    assert bucket.reserve(5000) == pytest.approx(2800 / 120)


def test_token_bucket_refills_with_time(clock):
    bucket = TokenBucket(7200)
    bucket.reserve(7200)
    clock.now += 30  # 3600 tokens are back
    assert bucket.reserve(3600) == pytest.approx(0.0)


def test_token_bucket_caps_an_oversized_request(clock):
    assert TokenBucket(1000).reserve(5000) == 0.0


def test_token_bucket_settle_returns_unused_tokens(clock):
    bucket = TokenBucket(7200)
    bucket.reserve(5000)
    bucket.settle(reserved=5000, actual=1000)
    assert bucket.reserve(6000) == 0.0


def test_token_bucket_drain_forces_a_wait(clock):
    bucket = TokenBucket(7200)
    bucket.drain()
    assert bucket.reserve(1200) == pytest.approx(10.0)


def test_invoke_records_stats_and_strips_reasoning(clock):
    t = throttled(FakeChat([ai("<think>x</think>Hello", 120, 30)]))
    assert t.invoke("hi").content == "Hello"
    assert llm_mod.STATS["calls"] == 1
    assert llm_mod.STATS["prompt_tokens"] == 120
    assert llm_mod.STATS["completion_tokens"] == 30


def test_rate_limit_waits_for_the_provider_hint_then_retries(clock):
    error = status_error(openai.RateLimitError, "Rate limit reached. Please try again in 2s.", 429)
    t = throttled(FakeChat([error, ai("ok")]))
    assert t.invoke("hi").content == "ok"
    assert clock.sleeps == [pytest.approx(3.0)]
    assert llm_mod.STATS["throttle_seconds"] == pytest.approx(3.0)


def test_daily_quota_stops_immediately_without_waiting(clock):
    error = status_error(openai.RateLimitError, "Limit reached on requests per day (RPD).", 429)
    with pytest.raises(RuntimeError, match="Daily quota exhausted"):
        throttled(FakeChat([error])).invoke("hi")
    assert clock.sleeps == []


def test_connection_errors_get_a_helpful_message(clock):
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://example.test"))
    with pytest.raises(RuntimeError, match="Cannot reach the LLM endpoint"):
        throttled(FakeChat([error])).invoke("hi")


def test_oversized_request_is_reported_clearly(clock):
    error = status_error(openai.APIStatusError, "Request too large", 413)
    with pytest.raises(RuntimeError, match="too large"):
        throttled(FakeChat([error])).invoke("hi")


def test_prompt_larger_than_the_token_budget_fails_before_calling(clock):
    chat = FakeChat([ai("never used")])
    with pytest.raises(RuntimeError, match="exceeds the per-minute token budget"):
        throttled(chat, bucket=TokenBucket(500)).invoke("x" * 10000)
    assert chat.calls == 0


def test_token_pacing_delays_the_second_call(clock):
    t = throttled(FakeChat([ai("one", 3000, 100), ai("two", 3000, 100)]), bucket=TokenBucket(7200))
    prompt = "x" * 16000  # about 4000 tokens plus the reply allowance
    t.invoke(prompt)
    t.invoke(prompt)
    assert llm_mod.STATS["throttle_seconds"] == pytest.approx(2.5)


def test_request_interval_spaces_calls(clock):
    t = throttled(FakeChat([ai("a"), ai("b")]), min_interval=4.0)
    t.invoke("p")
    t.invoke("p")
    assert clock.sleeps == [pytest.approx(4.0)]
