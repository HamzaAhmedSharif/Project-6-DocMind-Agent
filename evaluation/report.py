"""Scores a saved comparison run and writes the results table. No network, no LLM."""
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.questions import QUESTIONS
from evaluation.scoring import score_key_facts, is_abstention

DEFAULT_RUN = os.path.join("evaluation", "results", "run_main.json")
DEFAULT_MD = os.path.join("evaluation", "RESULTS.md")
DEFAULT_SCORES = os.path.join("evaluation", "results", "scores_main.json")
COST_FIELDS = ["calls", "prompt_tokens", "completion_tokens", "api_seconds", "throttle_seconds", "wall_seconds"]
SYSTEMS = ("baseline", "multi")
QMAP = {q["id"]: q for q in QUESTIONS}


def complete(entry, system):
    return system in entry and "error" not in entry[system]


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value, digits=1):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if digits else f"{value:.0f}"


def ratio(a, b):
    return f"{a}/{b}" if b else "n/a"


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def score_run(results):
    scores = {}
    for qid, entry in results.items():
        q = QMAP[qid]
        per_system = {}
        for system in SYSTEMS:
            if not complete(entry, system):
                continue
            r = entry[system]
            item = {"type": q["type"]}
            if q["type"] == "doc":
                item.update(score_key_facts(r["answer"], q["key_facts"]))
            else:
                item["abstained"] = is_abstention(r["answer"])
                if q["type"] == "web" and system == "multi":
                    item["routed_web"] = r["route"] == "WEB"
                    item["has_web_urls"] = len(r.get("web_urls", [])) > 0
            if system == "multi":
                item["route_correct"] = (r["route"] == "WEB") == (q["type"] == "web")
                item["revised"] = r.get("revisions", 0) >= 1
            per_system[system] = item
        scores[qid] = per_system
    return scores


def items(scores, system, qtype):
    return [v[system] for v in scores.values() if system in v and v[system]["type"] == qtype]


def cost_means(results, system, types):
    rs = [results[qid][system] for qid in results if QMAP[qid]["type"] in types and complete(results[qid], system)]
    return len(rs), {f: mean([r[f] for r in rs]) for f in COST_FIELDS}


def build_markdown(run, results, scores):
    fp = run["fingerprint"]["settings"]
    n_complete = sum(1 for e in results.values() for s in SYSTEMS if complete(e, s))
    n_error = sum(1 for e in results.values() for s in SYSTEMS if s in e and "error" in e[s])
    lines = [
        "# Comparison results: single-agent baseline vs multi-agent",
        "",
        f"Run started {run['started']}. Model `{fp['model']}` via `{fp['provider']}`, extra request fields {fp['extra_body']}. "
        f"Retrieval: {fp['chunk_size']}-word chunks, overlap {fp['chunk_overlap']}, top-{fp['top_k']}, "
        f"embedding model `{fp['embedding_model']}`. Temperature 0 for every call. "
        f"Complete results: {n_complete} of {2 * len(QUESTIONS)}; errors: {n_error}.",
        "",
        "Scoring is rule-based (`evaluation/scoring.py`); raw answers are in `evaluation/results/run_main.json`.",
        "",
        "## Answer quality",
        "",
        "| Metric | Single-agent baseline | Multi-agent |",
        "|---|---|---|",
    ]

    def doc_cell(system):
        its = items(scores, system, "doc")
        found = sum(i["found"] for i in its)
        total = sum(i["total"] for i in its)
        return f"{pct(found, total)} ({found}/{total} facts, {len(its)} questions)"

    def count_cell(system, qtype, key):
        its = items(scores, system, qtype)
        return ratio(sum(1 for i in its if i.get(key)), len(its))

    multi_all = [v["multi"] for v in scores.values() if "multi" in v]
    lines += [
        f"| Answerable questions: key-fact coverage | {doc_cell('baseline')} | {doc_cell('multi')} |",
        f"| Answerable questions: all key facts found | {count_cell('baseline', 'doc', 'all_found')} | {count_cell('multi', 'doc', 'all_found')} |",
        f"| Not-in-paper questions: abstained correctly | {count_cell('baseline', 'absent', 'abstained')} | {count_cell('multi', 'absent', 'abstained')} |",
        f"| Web questions: routed to web search | n/a | {count_cell('multi', 'web', 'routed_web')} |",
        f"| Web questions: answer cites web URLs | n/a | {count_cell('multi', 'web', 'has_web_urls')} |",
        f"| Web questions: said it could not answer | {count_cell('baseline', 'web', 'abstained')} | {count_cell('multi', 'web', 'abstained')} |",
        f"| Routing correct (all questions) | n/a | {ratio(sum(1 for i in multi_all if i['route_correct']), len(multi_all))} |",
        f"| Reviewer requested a revision | n/a | {ratio(sum(1 for i in multi_all if i['revised']), len(multi_all))} |",
        "",
        "## Cost per question (mean)",
        "",
        "| Metric | Baseline, no-web questions | Multi-agent, no-web questions | Baseline, web questions | Multi-agent, web questions |",
        "|---|---|---|---|---|",
    ]
    groups = [("baseline", ("doc", "absent")), ("multi", ("doc", "absent")), ("baseline", ("web",)), ("multi", ("web",))]
    stats = [cost_means(results, system, types) for system, types in groups]
    lines.append("| Questions measured | " + " | ".join(str(n) for n, _ in stats) + " |")
    for field in COST_FIELDS:
        digits = 0 if field.endswith("tokens") else 1
        lines.append(f"| {field} | " + " | ".join(fmt(m[field], digits) for _, m in stats) + " |")
    lines += [
        "",
        "`api_seconds` is time spent inside provider calls; `throttle_seconds` is time spent waiting to stay under the free-tier token limit; `wall_seconds` is both plus pipeline overhead. Compare systems on `api_seconds`, since waiting time depends on what ran just before.",
        "",
        "## Per-question detail",
        "",
        "| id | type | baseline | multi-agent | route | revised | multi calls | multi prompt tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def cell(qid, system):
        sc = scores.get(qid, {}).get(system)
        if sc is None:
            return "error/missing"
        if sc["type"] == "doc":
            return f"{sc['found']}/{sc['total']} facts"
        return "abstained" if sc["abstained"] else "answered"

    for q in QUESTIONS:
        qid = q["id"]
        entry = results.get(qid, {})
        multi = entry.get("multi", {}) if complete(entry, "multi") else {}
        lines.append(
            f"| {qid} | {q['type']} | {cell(qid, 'baseline')} | {cell(qid, 'multi')} | {multi.get('route', 'n/a')} | "
            f"{'yes' if multi.get('revisions', 0) >= 1 else 'no'} | {multi.get('calls', 'n/a')} | {multi.get('prompt_tokens', 'n/a')} |"
        )
    return "\n".join(lines) + "\n"


def audit(results, scores):
    """Print raw answer snippets for anything that was not a full pass, so scoring can be checked by eye."""
    for qid, entry in results.items():
        q = QMAP[qid]
        for system in SYSTEMS:
            if system in entry and "error" in entry[system]:
                print(f"[{qid}] {system}: ERROR {entry[system]['error']}")
                continue
            if not complete(entry, system):
                continue
            r = entry[system]
            sc = scores[qid][system]
            if q["type"] == "doc":
                if sc["all_found"]:
                    continue
                missed = [kf["fact"] for kf, hit in zip(q["key_facts"], sc["hits"]) if not hit]
                tag = f"missed: {missed}"
            else:
                tag = f"abstained={sc['abstained']}"
                if system == "multi":
                    tag += f" route={r['route']} web_urls={len(r.get('web_urls', []))}"
            print(f"[{qid}] {system} {tag}\n    {' '.join(r['answer'].split())[:350]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--md", default=DEFAULT_MD)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    with open(args.run, encoding="utf-8") as f:
        run = json.load(f)
    results = run["results"]
    scores = score_run(results)
    markdown = build_markdown(run, results, scores)

    for path in (args.md, args.scores):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(args.scores, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(markdown)
    if args.audit:
        print("--- AUDIT ---")
        audit(results, scores)


if __name__ == "__main__":
    main()
