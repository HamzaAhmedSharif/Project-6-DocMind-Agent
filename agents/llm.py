import sys
import os
import re
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import APIConnectionError, RateLimitError
from langchain_openai import ChatOpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_RPM, LLM_EXTRA_BODY, SUPERVISOR_MODEL, WRITER_MODEL

ROLES = {
    "supervisor": (SUPERVISOR_MODEL, 0.0),
    "writer": (WRITER_MODEL, 0.0),
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_last_call_time = {}  # request limits are per model, so pacing is tracked per model

STATS = {"calls": 0, "api_seconds": 0.0, "throttle_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def reset_stats():
    STATS.update({"calls": 0, "api_seconds": 0.0, "throttle_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0})


def _strip_reasoning(text):
    """Remove inline <think>...</think> blocks that some models put in `content`."""
    if isinstance(text, str):
        return _THINK_RE.sub("", text).strip()
    return text


class ThrottledLLM:
    """Chat client for any OpenAI-compatible endpoint, with optional pacing and rate-limit handling."""

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.min_interval = (60.0 / LLM_RPM + 1.0) if LLM_RPM > 0 else 0.0
        self.llm = ChatOpenAI(
            model=model,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=temperature,
            max_retries=0,
            extra_body=LLM_EXTRA_BODY or None,
        )

    def invoke(self, prompt: str):
        for attempt in range(1, 5):
            wait = self.min_interval - (time.time() - _last_call_time.get(self.model, 0.0))
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
                STATS["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                STATS["completion_tokens"] += usage.get("completion_tokens", 0) or 0
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
                message = str(e).lower()
                if "per day" in message or "daily" in message:
                    raise RuntimeError(
                        f"Daily quota exhausted for {self.model}. Switch model/provider or wait for the reset."
                    ) from e
                print(f"[LLM] Rate limited on {self.model}; waiting 60s (attempt {attempt}/4)")
                time.sleep(60)
                STATS["throttle_seconds"] += 60
        raise RuntimeError(f"Rate limit persisted on {self.model} after 4 attempts")


def get_llm(role: str) -> ThrottledLLM:
    model, temperature = ROLES[role]
    return ThrottledLLM(model, temperature)
