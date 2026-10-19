# Comparison results: single-agent baseline vs multi-agent

Run started 2026-10-19 09:32:33. Model `openai/gpt-oss-120b` via `hosted`, extra request fields {'reasoning_effort': 'low'}. Retrieval: 150-word chunks, overlap 25, top-10, embedding model `all-MiniLM-L6-v2`. Temperature 0 for every call. Complete results: 40 of 40; errors: 0.

Scoring is rule-based (`evaluation/scoring.py`); raw answers are in `evaluation/results/run_main.json`.

## Answer quality

| Metric | Single-agent baseline | Multi-agent |
|---|---|---|
| Answerable questions: key-fact coverage | 76.9% (20/26 facts, 12 questions) | 73.1% (19/26 facts, 12 questions) |
| Answerable questions: all key facts found | 7/12 | 5/12 |
| Not-in-paper questions: abstained correctly | 4/4 | 4/4 |
| Web questions: routed to web search | n/a | 4/4 |
| Web questions: answer cites web URLs | n/a | 4/4 |
| Web questions: said it could not answer | 3/4 | 0/4 |
| Routing correct (all questions) | n/a | 20/20 |
| Reviewer requested a revision | n/a | 2/20 |

## Cost per question (mean)

| Metric | Baseline, no-web questions | Multi-agent, no-web questions | Baseline, web questions | Multi-agent, web questions |
|---|---|---|---|---|
| Questions measured | 16 | 16 | 4 | 4 |
| calls | 1.0 | 3.1 | 1.0 | 3.5 |
| prompt_tokens | 2549 | 5856 | 2533 | 8991 |
| completion_tokens | 159 | 298 | 90 | 674 |
| api_seconds | 2.1 | 6.1 | 2.2 | 8.6 |
| throttle_seconds | 16.9 | 41.7 | 18.6 | 69.0 |
| wall_seconds | 19.1 | 47.9 | 20.9 | 81.7 |

`api_seconds` is time spent inside provider calls; `throttle_seconds` is time spent waiting to stay under the free-tier token limit; `wall_seconds` is both plus pipeline overhead. Compare systems on `api_seconds`, since waiting time depends on what ran just before.

## Per-question detail

| id | type | baseline | multi-agent | route | revised | multi calls | multi prompt tokens |
|---|---|---|---|---|---|---|---|
| d01 | doc | 2/2 facts | 1/2 facts | DOC | no | 3 | 4831 |
| d02 | doc | 2/3 facts | 2/3 facts | DOC | no | 3 | 5079 |
| d03 | doc | 2/2 facts | 2/2 facts | DOC | no | 3 | 5574 |
| d04 | doc | 1/2 facts | 1/2 facts | DOC | no | 3 | 5228 |
| d05 | doc | 2/2 facts | 2/2 facts | DOC | no | 3 | 5424 |
| d06 | doc | 2/2 facts | 1/2 facts | DOC | no | 3 | 5250 |
| d07 | doc | 2/3 facts | 3/3 facts | DOC | no | 3 | 5580 |
| d08 | doc | 2/2 facts | 1/2 facts | DOC | no | 3 | 5848 |
| d09 | doc | 1/2 facts | 1/2 facts | DOC | yes | 5 | 12336 |
| d10 | doc | 2/2 facts | 2/2 facts | DOC | no | 3 | 5813 |
| d11 | doc | 2/2 facts | 2/2 facts | DOC | no | 3 | 5096 |
| d12 | doc | 0/2 facts | 1/2 facts | DOC | no | 3 | 4780 |
| w01 | web | abstained | answered | WEB | no | 3 | 7569 |
| w02 | web | answered | answered | WEB | no | 3 | 7490 |
| w03 | web | abstained | answered | WEB | no | 3 | 6773 |
| w04 | web | abstained | answered | WEB | yes | 5 | 14131 |
| a01 | absent | abstained | abstained | DOC | no | 3 | 5826 |
| a02 | absent | abstained | abstained | DOC | no | 3 | 5995 |
| a03 | absent | abstained | abstained | DOC | no | 3 | 5679 |
| a04 | absent | abstained | abstained | DOC | no | 3 | 5361 |
