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

## M3b prep - pacing and evaluation design (2026-10-18)

### Note: the binding limit was tokens per minute, and a fixed 60-second sleep wasted time
- Symptom: each multi-agent test query hit one rate-limit retry and waited a fixed 60 seconds.
- Isolation: the provider headers reported `x-ratelimit-limit-tokens: 8000` (per minute) and `x-ratelimit-limit-requests: 1000` (per day). Measured prompt tokens per query were 4198 (baseline, 1 call), 8743 (document question, 3 calls) and 13048 (web question, 3 calls), so one multi-agent query exceeds a full minute of token budget while requests stay far below their limit. The headers also showed the budget refilling continuously (590 tokens used, reset in 4.425 s, about 133 tokens/s) and that the limiter counted the prompt plus the requested `max_tokens` (78 + 512 = 590 deducted for a 97-token call).
- Cause: pacing by request count and a blind 60-second sleep ignored the token budget.
- Fix: `agents/llm.py` now paces with a token bucket per model (`HOSTED_TPM` in `.env`), reads the provider's `try again in` hint for retries, fails with a clear message on requests larger than the per-minute budget, and does not set `max_tokens` on pipeline calls. Measured after the change on the same document question: throttle_seconds=17.466099087397257, rate-limit retries=0 (before: one 60-second retry).
- Lesson: read which limit the provider actually enforces, and pace against that one.

### Decision: rule-based scoring for the comparison, not an LLM judge
- Context: the comparison needs answer quality for two systems on the same questions, and the free tier allows about 8000 tokens per minute.
- Chosen: key-fact coverage (regular expressions) for questions the paper answers, routing correctness plus citation presence for questions that need the web, and abstention for questions the paper does not answer. Every key fact is checked against the paper's own text by `evaluation/validate_questions.py` (26 of 26 facts found). Retrieval is also checked offline: whether the retrieved context contains the evidence for each key fact (`evaluation/retrieval_grid.py`).
- Rejected: LLM-as-judge (the model that wrote an answer would grade it leniently, and judge calls would consume the same token budget).
- Tradeoff accepted: regular-expression scoring can miss a correct answer that is phrased unusually; per-question outputs are committed so every scoring call can be audited.
- Would revisit if: manual review of the committed outputs shows frequent scoring misses.

### Decision: chunk size and top-k chosen by retrieval-only measurement, before any answer was scored
- Context: with the original settings (800-word chunks, top-4) the retrieved context contained the evidence for only 61.5% of the 26 key facts (16 of 26) at about 3,930 context tokens, and one multi-agent query used 8,743 prompt tokens against a limit of 8,000 tokens per minute.
- Chosen: chunk size 150 words, overlap 25, top-10, selected by a rule fixed in advance (`select_config` in `evaluation/retrieval_grid.py`): among configurations with at most 2,200 context tokens, the highest fact recall, ties to fewer tokens, then smaller chunks. Evidence: `evaluation/results/retrieval_grid.json`; recall 92.3% at about 1904 context tokens.
- Rejected: 800-word chunks with top-5 (96.2% recall, but about 4,912 context tokens, which leaves no headroom under the 8,000 tokens/min limit once web results, measured at about 2,150 tokens per call, are added).
- Tradeoff accepted: the settings were selected on the same 12 questions later used to score answers, so reported context recall is optimistic; smaller chunks carry less surrounding text. The answer comparison itself is unaffected, because both systems use the same retrieval.
- Would revisit if: a held-out question set shows lower recall.

## M3c - scoring correction and ablation (2026-10-19)

### Note: the scorer misread typographic characters
- Symptom: the first report showed the baseline ahead on key-fact coverage (20/26 vs 19/26).
- Isolation: auditing the raw answers behind the missed facts showed correct answers scored as misses: they use non-breaking hyphens (`Scaled Dot-Product Attention`, `feed-forward`) and a spaced thousands separator (`4 000 steps`), which the regular expressions did not match.
- Cause: `evaluation/scoring.py` compared raw text without normalizing Unicode hyphens, spaces and thousands separators.
- Fix: answers are normalized (Unicode NFKC, hyphen and space variants, quotes, thousands separators) before matching, for every system alike. No expected-answer pattern was loosened, so correct paraphrases still count as misses. Rescoring the saved raw outputs (no new LLM calls) changed 5 fact counts (d02 baseline and multi-agent, d04 baseline and multi-agent, d08 multi-agent); coverage became baseline 22/26 and multi-agent 22/26. The first report remains in git history (commit 558bdb0).
- Lesson: audit the raw outputs behind a metric before reading a small difference as real; with 26 facts, one miss is 3.8 points.

### Decision: add a single-call ablation that uses the multi-agent writer prompt
- Context: the baseline (M2) prompt says to answer only from the context and to say so when the context lacks the answer; the multi-agent writer prompt does not, so differences between the two systems could come from prompt wording instead of routing, review and web search.
- Chosen: a third system, `single_writer` (retrieve, then one call with the multi-agent writer prompt, no routing, review or web search), run on the same 20 questions after the first run; its results were added to the same run file and the first-run results are unchanged. Evidence: key-fact coverage baseline 22/26, single call 23/26, multi-agent 22/26.
- Rejected: reading the baseline-vs-multi-agent gap as an effect of agent structure alone.
- Tradeoff accepted: 20 more LLM calls and a comparison added after seeing the first results; it is disclosed here and nothing was changed to improve any system.
- Would revisit if: more questions are added, since 12 answerable questions are too few to separate small differences.

### Run note: provider credential replaced mid-run
- The provider began rejecting the original API key with HTTP 403 partway through the ablation, and the key was replaced in `.env`. The resume check fingerprints the pipeline code and settings, not credentials, so the run continued without mixing results.

## M3d - evaluation finalization (2026-10-19)

### Note: the web "cites URLs" metric measured retrieval, not citation
- Symptom: the report said the multi-agent answers cited web URLs on 4 of 4 web questions.
- Isolation: reading the saved answers showed no URL in the answer text of any of them (one answer even has an empty "Sources" column); the metric only checked that web results had been retrieved.
- Cause: `report.py` tested whether the pipeline state held web URLs and labelled that as citation.
- Fix: the row is now "web results retrieved" (4/4), and a separate row counts answers whose text contains a URL (3/4).
- Lesson: name a metric for what it measures; a label that sounds stronger than the check is a bug.

### Decision: paraphrase review reported next to the strict score
- Context: strict regular-expression scoring counts a correct paraphrase as a miss, and 8 of the 11 strict misses were correct answers in different words (for example "each position ... can attend to ... all other positions in that same sequence").
- Chosen: keep the strict score as the headline and add a second row, "coverage after paraphrase review", from `evaluation/adjudication.json`: 8 entries, each with a quote that the report script verifies against the saved answer. Evidence: strict coverage baseline 22/26, single call 23/26, multi-agent 22/26; after review baseline 24/26, single call 26/26, multi-agent 25/26.
- Rejected: loosening the expected-answer patterns after seeing the results (that tunes the scorer toward the outcome); an LLM judge (same-model leniency, and extra tokens under the free-tier limit).
- Tradeoff accepted: the entries were proposed by the project's AI assistant and are not yet confirmed by the author, which the results table says; with 26 facts the two views differ by only a few facts and no system is clearly better.
- Would revisit if: the author's review changes an entry, or more questions are added.

### Note: some correct answers may come from the model's memory, not the retrieved text
- Symptom: the single-call system stated "label smoothing" for the regularization question although the retrieved context for that question did not contain it.
- Isolation: `evaluation/grounding_check.py` compares each stated key fact with the retrieved context (no LLM calls). Key facts whose evidence is absent from the retrieved context: 2 of 26. Facts stated although their evidence was absent from the context: baseline 1, single call 2, multi-agent 1.
- Cause: the paper is famous, and the writer prompt does not require answers to come only from the context, so a model can answer from memory; the baseline prompt does require it.
- Consequence: key-fact coverage cannot tell a grounded answer from a remembered one on this document. A document published after the model's training cutoff, or a private one, is needed for a clean test (planned for the later evaluation project).
- Lesson: choose evaluation documents the model cannot already know.

### Note: reviewer feedback leaked into final answers
- Symptom: the multi-agent final answer for w04 ends with "addressing the reviewer's concern"; questions whose multi-agent answer mentions the reviewer: w04.
- Cause: on a revision the writer prompt contains the reviewer's feedback as an instruction ("address it"), and the model answered the reviewer inside the user-facing text.
- Status: not fixed, because the pipeline code was frozen for the evaluation. The fix is one sentence in the writer's feedback note ("do not mention the reviewer or this feedback"), followed by a rerun of the comparison.
- Lesson: text meant for the model can surface in the output; read the raw answers, not only the scores.

## M5 - UI (2026-10-19)

### Decision: the UI is a thin layer over the evaluated pipeline
- Context: the pipeline code and settings were frozen when the comparison ran; a UI that uploads and indexes arbitrary PDFs would need changes inside `agents/`.
- Chosen: `app.py` only calls the existing functions on the index built by `agents/ingest.py`, and shows routing, sources, call count, tokens and rate-limit waiting beside each answer. Evidence: `scripts/smoke_app.py` (headless AppTest passed: the app loads and baseline mode answered with exactly one LLM call).
- Rejected: refactoring `Retriever` for in-memory uploads now (the evaluated code would then differ from the shipped code).
- Tradeoff accepted: the demo answers questions about the indexed document only; uploads and a service API belong to the productionization project.
- Would revisit if: the evaluation is rerun on new code.
