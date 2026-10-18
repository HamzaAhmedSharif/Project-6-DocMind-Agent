import sys
import os
import re
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import APIConnectionError, APIStatusError, RateLimitError
from langchain_openai import ChatOpenAI
from config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_RPM, LLM_TPM, LLM_EXTRA_BODY,
    SUPERVISOR_MODEL, WRITER_MODEL,
)

ROLES = {
    "supervisor": (SUPERVISOR_MODEL, 0.0),
    "writer": (WRITER_MODEL, 0.0),
}

CHARS_PER_TOKEN = 4.0       # rough estimate, used only to pace requests (real prompts run about 4.5-4.8)
COMPLETION_ALLOWANCE = 400  # tokens reserved for the reply when pacing
SAFETY_FRACTION = 0.9       # use at most 90% of the provider's per-minute token budget

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RETRY_RE = re.compile(r"try again in\s+([0-9hms.\s]+)", re.IGNORECASE)
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_last_call_time = {}  # model -> time of last request (request limits are per model)
_buckets = {}         # model -> TokenBucket (token limits are per model)

STATS = {"calls": 0, "api_seconds": 0.0, "throttle_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def reset_stats():
    STATS.update({"calls": 0, "api_seconds": 0.0, "throttle_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0})


def estimate_tokens(prompt: str) -> int:
    return int(len(prompt) / CHARS_PER_TOKEN) + COMPLETION_ALLOWANCE


def parse_retry_after(message: str, default: float = 60.0) -> float:
    """Seconds to wait according to a provider message such as 'try again in 1m26.4s' (plus a 1s margin)."""
    match = _RETRY_RE.search(message)
    if not match:
        return default
    total = sum(float(v) * _UNIT_SECONDS[u] for v, u in _UNIT_RE.findall(match.group(1)))
    return total + 1.0 if total > 0 else default


def _strip_reasoning(text):
    """Remove inline <think>...</think> blocks that some models put in `content`."""
    if isinstance(text, str):
        return _THINK_RE.sub("", text).strip()
    return text


class TokenBucket:
    """Continuously refilling token budget; the provider's per-minute limit behaves this way."""

    def __init__(self, tokens_per_minute: int):
        self.capacity = float(tokens_per_minute)
        self.rate = self.capacity / 60.0
        self.level = self.capacity
        self.last = time.time()

    def _refill(self, now: float):
        self.level = min(self.capacity, self.level + (now - self.last) * self.rate)
        self.last = now

    def reserve(self, tokens: int) -> float:
        """Reserve tokens and return how many seconds to wait before sending."""
        self._refill(time.time())
        tokens = min(float(tokens), self.capacity)
        wait = max(0.0, (tokens - self.level) / self.rate)
        self.level -= tokens  # may go negative: a debt that refills while we wait
        return wait

    def settle(self, reserved: int, actual: int):
        """Replace the estimate with the real token count after a successful call."""
        self._refill(time.time())
        self.level = min(self.capacity, self.level + min(float(reserved), self.capacity) - float(actual))

    def drain(self):
        """The provider said we are out of budget: assume empty."""
        self._refill(time.time())
        self.level = 0.0


class ThrottledLLM:
    """Chat client for any OpenAI-compatible endpoint, with request/token pacing and rate-limit handling."""

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.min_interval = (60.0 / LLM_RPM + 1.0) if LLM_RPM > 0 else 0.0
        if LLM_TPM > 0:
            self.bucket = _buckets.setdefault(model, TokenBucket(int(LLM_TPM * SAFETY_FRACTION)))
        else:
            self.bucket = None
        self.llm = ChatOpenAI(
            model=model,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=temperature,
            max_retries=0,
            extra_body=LLM_EXTRA_BODY or None,
        )

    def invoke(self, prompt: str):
        reserved = estimate_tokens(prompt)
        if self.bucket is not None and reserved > self.bucket.capacity:
            raise RuntimeError(
                f"Prompt (~{reserved} tokens) exceeds the per-minute token budget "
                f"({int(self.bucket.capacity)}). Reduce TOP_K_RESULTS or CHUNK_SIZE."
            )
        for attempt in range(1, 5):
            wait = self.min_interval - (time.time() - _last_call_time.get(self.model, 0.0))
            if self.bucket is not None:
                wait = max(wait, self.bucket.reserve(reserved))
            if wait > 0:
                time.sleep(wait)
                STATS["throttle_seconds"] += wait
            _last_call_time[self.model] = time.time()
            start = time.time()
            try:
                response = self.llm.invoke(prompt)
                STATS["calls"] += 1
                STATS["api_seconds"] += time.time() - start
                usage = (response.response_metadata or {}).get("token_usage") or {}
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0
                STATS["prompt_tokens"] += prompt_tokens
                STATS["completion_tokens"] += completion_tokens
                if self.bucket is not None and (prompt_tokens + completion_tokens) > 0:
                    self.bucket.settle(reserved, prompt_tokens + completion_tokens)
                response.content = _strip_reasoning(response.content)
                if not response.content:
                    print(f"[LLM] Warning: empty reply from {self.model}")
                return response
            except APIConnectionError as e:
                raise RuntimeError(
                    f"Cannot reach the LLM endpoint at {LLM_BASE_URL}. "
                    "Hosted provider: check the base URL and that your network can reach it. "
                    "Ollama: open the Ollama app (or run `ollama serve`) and check `ollama list`."
                ) from e
            except RateLimitError as e:
                message = str(e)
                if "per day" in message.lower() or "daily" in message.lower():
                    raise RuntimeError(
                        f"Daily quota exhausted for {self.model}. Switch model/provider or wait for the reset."
                    ) from e
                if self.bucket is not None:
                    self.bucket.drain()
                delay = parse_retry_after(message)
                print(f"[LLM] Rate limited on {self.model}; waiting {delay:.0f}s (attempt {attempt}/4)")
                time.sleep(delay)
                STATS["throttle_seconds"] += delay
            except APIStatusError as e:
                if e.status_code == 413:
                    raise RuntimeError(
                        "Request too large for the provider's per-minute token limit. "
                        "Reduce TOP_K_RESULTS or CHUNK_SIZE."
                    ) from e
                raise
        raise RuntimeError(f"Rate limit persisted on {self.model} after 4 attempts")


def get_llm(role: str) -> ThrottledLLM:
    model, temperature = ROLES[role]
    return ThrottledLLM(model, temperature)
