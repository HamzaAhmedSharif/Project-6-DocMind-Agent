"""Connectivity and rate-limit probe for the configured LLM backend. Never prints credentials."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_EXTRA_BODY


def main():
    print(f"provider={LLM_PROVIDER} base_url={LLM_BASE_URL} model={LLM_MODEL} extra_body={LLM_EXTRA_BODY}")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, max_retries=0)
    extra = {"extra_body": LLM_EXTRA_BODY} if LLM_EXTRA_BODY else {}
    raw = client.chat.completions.with_raw_response.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=512,
        temperature=0,
        **extra,
    )
    completion = raw.parse()
    choice = completion.choices[0]
    print("status:", raw.http_response.status_code)
    print("reply:", repr((choice.message.content or "").strip()))
    print("finish_reason:", choice.finish_reason)
    print("usage:", completion.usage)
    for name, value in raw.headers.items():
        if name.lower().startswith("x-ratelimit") or name.lower() == "retry-after":
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
