# Comparison results: baseline vs single call vs multi-agent

Run started 2026-10-19 09:32:33. Model `openai/gpt-oss-120b` via `hosted`, extra request fields {'reasoning_effort': 'low'}. Retrieval: 150-word chunks, overlap 25, top-10, embedding model `all-MiniLM-L6-v2`. Temperature 0 for every call. Complete results: 60 of 60; errors: 0.

Systems: **baseline** = the M2 pipeline (one call, its original prompt, which says to answer only from the context and to say so when it is missing); **single call, writer prompt** = the same retrieval and one call using the multi-agent writer prompt, with no routing, review or web search (separates the effect of orchestration from the effect of prompt wording); **multi-agent** = supervisor routing, researcher (document retrieval plus web search when routed), writer, and supervisor review with at most one revision.

Scoring is rule-based (`evaluation/scoring.py`). A fact counts only if the answer contains the expected wording after normalizing Unicode dashes, spaces and thousands separators; correct paraphrases count as misses for every system in the strict rows (the paraphrase-review row counts them separately). Web questions are scored only on routing, on whether web results were retrieved and on whether the answer text contains a URL; the correctness of web answers is not measured here. Raw answers are in `evaluation/results/run_main.json`.

## Answer quality

| Metric | Baseline (M2 prompt, 1 call) | Single call, writer prompt | Multi-agent |
|---|---|---|---|
| Answerable questions: key-fact coverage | 84.6% (22/26) | 88.5% (23/26) | 84.6% (22/26) |
| Answerable questions: coverage after paraphrase review (see the Paraphrase review section) | 92.3% (24/26) | 100.0% (26/26) | 96.2% (25/26) |
| Answerable questions: all key facts found | 9/12 | 9/12 | 8/12 |
| Answerable questions: multi-agent found more / same / fewer facts than this system (questions) | 2 / 8 / 2 | 0 / 11 / 1 | n/a |
| Not-in-paper questions: abstained | 4/4 | 4/4 | 4/4 |
| Web questions: said it could not answer | 3/4 | 3/4 | 0/4 |
| Web questions: routed to web search | n/a | n/a | 4/4 |
| Web questions: web results retrieved | n/a | n/a | 4/4 |
| Web questions: answer text contains a URL | n/a | n/a | 3/4 |
| Routing correct (all questions) | n/a | n/a | 20/20 |
| Reviewer requested a revision | n/a | n/a | 2/20 |

## Cost per question (mean): questions that need no web search

| Metric | Baseline (M2 prompt, 1 call) | Single call, writer prompt | Multi-agent |
|---|---|---|---|
| Questions measured | 16 | 16 | 16 |
| calls | 1.0 | 1.0 | 3.1 |
| prompt_tokens | 2549 | 2559 | 5856 |
| completion_tokens | 159 | 197 | 298 |
| api_seconds | 2.1 | 1.7 | 6.1 |
| throttle_seconds | 16.9 | 17.6 | 41.7 |
| wall_seconds | 19.1 | 19.3 | 47.9 |

## Cost per question (mean): questions that need web search

| Metric | Baseline (M2 prompt, 1 call) | Single call, writer prompt | Multi-agent |
|---|---|---|---|
| Questions measured | 4 | 4 | 4 |
| calls | 1.0 | 1.0 | 3.5 |
| prompt_tokens | 2533 | 2543 | 8991 |
| completion_tokens | 90 | 104 | 674 |
| api_seconds | 2.2 | 1.5 | 8.6 |
| throttle_seconds | 18.6 | 20.7 | 69.0 |
| wall_seconds | 20.9 | 22.3 | 81.7 |

`api_seconds` is time spent inside provider calls; `throttle_seconds` is time spent waiting to stay under the free-tier token limit; `wall_seconds` is both plus pipeline overhead. Compare systems on `api_seconds`, since waiting time depends on what ran just before.

## Per-question detail

| id | type | baseline | single call | multi-agent | route | revised | multi calls | multi prompt tokens |
|---|---|---|---|---|---|---|---|---|
| d01 | doc | 2/2 facts | 1/2 facts | 1/2 facts | DOC | no | 3 | 4831 |
| d02 | doc | 3/3 facts | 3/3 facts | 3/3 facts | DOC | no | 3 | 5079 |
| d03 | doc | 2/2 facts | 2/2 facts | 2/2 facts | DOC | no | 3 | 5574 |
| d04 | doc | 2/2 facts | 2/2 facts | 2/2 facts | DOC | no | 3 | 5228 |
| d05 | doc | 2/2 facts | 2/2 facts | 2/2 facts | DOC | no | 3 | 5424 |
| d06 | doc | 2/2 facts | 1/2 facts | 1/2 facts | DOC | no | 3 | 5250 |
| d07 | doc | 2/3 facts | 3/3 facts | 3/3 facts | DOC | no | 3 | 5580 |
| d08 | doc | 2/2 facts | 2/2 facts | 2/2 facts | DOC | no | 3 | 5848 |
| d09 | doc | 1/2 facts | 2/2 facts | 1/2 facts | DOC | yes | 5 | 12336 |
| d10 | doc | 2/2 facts | 2/2 facts | 2/2 facts | DOC | no | 3 | 5813 |
| d11 | doc | 2/2 facts | 2/2 facts | 2/2 facts | DOC | no | 3 | 5096 |
| d12 | doc | 0/2 facts | 1/2 facts | 1/2 facts | DOC | no | 3 | 4780 |
| w01 | web | abstained | abstained | answered | WEB | no | 3 | 7569 |
| w02 | web | answered | answered | answered | WEB | no | 3 | 7490 |
| w03 | web | abstained | abstained | answered | WEB | no | 3 | 6773 |
| w04 | web | abstained | abstained | answered | WEB | yes | 5 | 14131 |
| a01 | absent | abstained | abstained | abstained | DOC | no | 3 | 5826 |
| a02 | absent | abstained | abstained | abstained | DOC | no | 3 | 5995 |
| a03 | absent | abstained | abstained | abstained | DOC | no | 3 | 5679 |
| a04 | absent | abstained | abstained | abstained | DOC | no | 3 | 5361 |

## Paraphrase review

Strict scoring counts a correct paraphrase as a miss. The entries below are strict misses whose saved answer states the fact in different words; each quote is verified against the saved answer by the report script. The entries were proposed by the project's AI assistant and are to be confirmed by the author.

| id | system | fact | supporting quote |
|---|---|---|---|
| d01 | Single call, writer prompt | relates different positions of a single sequence | each position of the sequence can attend to (i.e., compute a weighted combination of) all other positions in that same sequence |
| d01 | Multi-agent | relates different positions of a single sequence | each position of the sequence can attend to (i.e., compute a weighted combination of) all other positions of that same sequence |
| d06 | Single call, writer prompt | attends to different representation subspaces | Capture information from multiple representation sub-spaces |
| d06 | Multi-agent | attends to different representation subspaces | Capture information from multiple representation sub-spaces |
| d12 | Baseline (M2 prompt, 1 call) | positions cannot attend to subsequent positions | each position cannot attend to any later (future) positions |
| d12 | Baseline (M2 prompt, 1 call) | predictions depend only on known outputs at earlier positions | the model is forced to use only the already-generated tokens when predicting the next token |
| d12 | Single call, writer prompt | positions cannot attend to subsequent positions | block each position from looking ahead to tokens that have not yet been generated |
| d12 | Multi-agent | positions cannot attend to subsequent positions | block each position from looking ahead to tokens that have not yet been generated |
