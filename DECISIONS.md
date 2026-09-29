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