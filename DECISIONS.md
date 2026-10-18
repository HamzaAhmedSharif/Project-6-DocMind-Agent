# DocMind Agent — Decisions & Engineering Notes

Decisions are logged as they are made, with the evidence available at the time.

## Decisions

### Decision: LLM backend for generation
- Context: the OpenAI API call failed with `429 insufficient_quota` (no prepaid credit); project budget is zero.
- Chosen: Google Gemini free tier via AI Studio. Evidence: `tests/test_m2.py` returns a grounded answer with a source citation (M2 commit).
- Rejected: OpenAI gpt-4o-mini (needs prepaid credit); local flan-t5-base as in Project 5 (assumed too weak for supervisor routing in M3, untested assumption).
- Tradeoff accepted: free-tier rate limits, and model names get retired. `gemini-2.0-flash` returned 404 mid-build and had to be bumped to `gemini-3.8-flash`.
- Would revisit if: rate limits throttle the M3 comparison runs or the Project 9 evaluation.

### Decision: shared `rag` conda environment with pinned requirements
- Context: `rag` already existed with newer package versions installed.
- Chosen: install the pinned `requirements.txt` into `rag`. Evidence: the pip log shows downgrades (numpy 2.2.6 to 1.26.4, transformers 5.16.1 to 4.57.6, langchain 1.3.18 to 0.3.7, langgraph 1.2.11 to 0.2.39, streamlit 1.63.0 to 1.39.0, faiss-cpu 1.15.0 to 1.9.0).
- Rejected: a dedicated environment for this project.
- Tradeoff accepted: other projects using `rag` now run against older versions; pip reported dependency conflicts for `langchain-classic` and `langgraph-prebuilt`.
- Would revisit if: any earlier project breaks under the downgraded packages. Then this project moves to its own environment.

## Engineering Notes

### `os.getenv()` called with the key value instead of the variable name
- Symptom: `ValueError: GOOGLE_API_KEY not set` while `.env` contained a key.
- Isolation: reading `.env` directly with `load_dotenv` returned the key, but `from config import GOOGLE_API_KEY` failed, so the fault was in `config.py`, not the file or its encoding.
- Cause: line 7 passed the key value as the argument to `os.getenv()`, so Python looked up an environment variable with that name and got `None`. How the line changed was not established.
- Fix: `os.getenv("GOOGLE_API_KEY")`.
- Lesson: test the loading layer and the consuming layer separately before suspecting file encoding.

### GitHub push protection blocked a commit containing a real API key
- Symptom: `GH013` on push; the key was on line 1 of `.env.example`.
- Cause: the real key had overwritten the placeholder in the template file.
- Fix: restored the placeholders, amended the unpushed commit (`git commit --amend --no-edit`), pushed, and rotated the key. The commit containing the key was never accepted by the remote.
- Lesson: template files can leak secrets too; the scanner caught it before publication.

## M3a - Groq backend and review fixes (2026-10-18)

### Decision: LLM backend - Groq free tier behind an OpenAI-compatible layer
- Context: the Gemini free tier (gemini-3.8-flash) returned 429 errors: first a limit of 5 requests/minute, then a cap of 20 requests per day per model. One multi-agent query needs 3-5 LLM calls and the comparison evaluation needs 60-100, so the cap made the pipeline unusable. This supersedes the earlier entry "LLM backend for generation".
- Chosen: Groq's free tier through its OpenAI-compatible endpoint, model `openai/gpt-oss-120b`. The backend is selected by `LLM_PROVIDER` in `.env`, so changing provider needs no code change. Evidence: `scripts/check_llm.py` returned HTTP 200; provider-reported limits: x-ratelimit-limit-requests=1000, x-ratelimit-limit-tokens=8000, x-ratelimit-remaining-requests=999, x-ratelimit-remaining-tokens=7410, x-ratelimit-reset-requests=1m26.4s, x-ratelimit-reset-tokens=4.425s.
- Rejected: staying on the Gemini free tier (daily cap); paid APIs (zero-cost constraint); local-only Ollama (an 8B model is weaker, and reviewers cannot be expected to run a GPU model). Ollama stays as an optional offline mode.
- Tradeoff accepted: depends on a third-party free tier whose limits can change (they already did once), so evaluation results are committed as files rather than expected to reproduce on demand.
- Would revisit if: the provider's limits fall below what the 20-question evaluation needs per day.

### Decision: the reviewer sees the retrieved context, not only the draft
- Context: the first version of the supervisor's review step received only the question and the draft answer, but was asked to judge whether the answer was grounded in the context.
- Chosen: pass the same formatted document and web context to the reviewer (`format_context` in `agents/writer.py`). Evidence: the diff of `agents/supervisor.py` and `agents/writer.py` in this commit. A reviewer without the evidence can only judge fluency.
- Rejected: leaving the reviewer blind (cheaper, but the verification step would verify nothing).
- Tradeoff accepted: the review prompt is about as large as the writer prompt, so tokens per query rise; the comparison reports tokens per query.
- Would revisit if: token limits make the evaluation infeasible; routing and review could then use separate models.

### Decision: model - openai/gpt-oss-120b with low reasoning effort
- Context: the models available to the free key were: allam-2-7b, canopylabs/orpheus-arabic-saudi, canopylabs/orpheus-v1-english, meta-llama/llama-prompt-guard-2-22m, meta-llama/llama-prompt-guard-2-86m, openai/gpt-oss-120b, openai/gpt-oss-20b, openai/gpt-oss-safeguard-20b, qwen/qwen3.8-27b, whisper-large-v3, whisper-large-v3-turbo. No large non-reasoning chat model was available; the 7B Arabic-focused model and the guard, speech and TTS models were unsuitable.
- Chosen: openai/gpt-oss-120b, the largest general model on the list, with the provider request field `reasoning_effort` set to low (`HOSTED_EXTRA_BODY` in `.env`) to limit tokens and latency. Evidence: `scripts/check_llm.py` returned reply 'ok' with finish_reason stop and usage CompletionUsage(completion_tokens=19, prompt_tokens=78, total_tokens=97, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=9, rejected_prediction_tokens=None), prompt_tokens_details=None, queue_time=0.006855616, prompt_time=0.003453878, completion_time=0.039720819, total_time=0.043174697).
- Rejected: the 7B model (weakest answers); the 20B model (smaller than the chosen one); the Qwen 27B model (kept as the fallback).
- Tradeoff accepted: reasoning models spend completion tokens on reasoning, so tokens and latency per call are higher than for a plain instruct model; the comparison reports both.
- Would revisit if: reasoning text leaks into `content`, replies come back empty, or token limits make the evaluation infeasible.

### Note: pacing code did not help against a daily quota
- Symptom: after adding client-side pacing for the 5 requests/minute limit, runs still failed with 429 on every retry.
- Isolation: the error text changed from `GenerateRequestsPerMinutePerProjectPerModel` (limit 5) to `GenerateRequestsPerDayPerProjectPerModel` (limit 20) while the pipeline waited and retried.
- Cause: a daily request cap on the model; waiting 60 seconds cannot clear it.
- Fix: the LLM layer now stops immediately with a clear message when the error mentions a daily limit (`agents/llm.py`), and the backend moved to a provider with a larger budget.
- Lesson: read which quota fired before adding retry logic; retries only help per-minute limits.

### Note: a retired model id broke the first generation call
- Symptom: `404 ... gemini-2.0-flash is no longer available`; the error message named the replacement model.
- Cause: the provider retired the model and the id was hard-coded.
- Fix: model ids moved into configuration (`config.py`, `.env`), so one line changes.
- Lesson: treat model ids as configuration, not constants.
