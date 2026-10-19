"""Scores a saved comparison run and writes the results table. No network, no LLM."""
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.adjudication import apply as apply_review, load_entries
from evaluation.questions import QUESTIONS
from evaluation.scoring import score_key_facts, is_abstention

DEFAULT_RUN = os.path.join("evaluation", "results", "run_main.json")
DEFAULT_MD = os.path.join("evaluation", "RESULTS.md")
DEFAULT_SCORES = os.path.join("evaluation", "results", "scores_main.json")
COST_FIELDS = ["calls", "prompt_tokens", "completion_tokens", "api_seconds", "throttle_seconds", "wall_seconds"]
SYSTEMS = ("baseline", "single_writer", "multi")
LABELS = {
    "baseline": "Baseline (M2 prompt, 1 call)",
    "single_writer": "Single call, writer prompt",
    "multi": "Multi-agent",
}
QMAP = {q["id"]: q for q in QUESTIONS}


def complete(entry, system):
    return system in entry and "error" not in entry[system]


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


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
            if system == "multi":
                item["routed_web"] = r["route"] == "WEB"
                item["web_results_retrieved"] = len(r.get("web_urls", [])) > 0
                item["url_in_answer"] = "http" in r["answer"].lower()
                item["route_correct"] = (r["route"] == "WEB") == (q["type"] == "web")
                item["revised"] = r.get("revisions", 0) >= 1
            per_system[system] = item
        scores[qid] = per_system
    return scores


def items(scores, system, qtype):
    return [v[system] for v in scores.values() if system in v and v[system]["type"] == qtype]


def paired(scores, a, b):
    """On answerable questions: how often system a found more / the same / fewer key facts than system b."""
    better = tie = worse = 0
    for qid, per in scores.items():
        if QMAP[qid]["type"] != "doc" or a not in per or b not in per:
            continue
        diff = per[a]["found"] - per[b]["found"]
        if diff > 0:
            better += 1
        elif diff == 0:
            tie += 1
        else:
            worse += 1
    return better, tie, worse


def cost_means(results, system, types):
    rs = [results[qid][system] for qid in results if QMAP[qid]["type"] in types and complete(results[qid], system)]
    return len(rs), {f: mean([r[f] for r in rs]) for f in COST_FIELDS}


def build_markdown(run, results, scores):
    fp = run["fingerprint"]["settings"]
    n_complete = sum(1 for e in results.values() for s in SYSTEMS if complete(e, s))
    n_error = sum(1 for e in results.values() for s in SYSTEMS if s in e and "error" in e[s])
    head = "| Metric | " + " | ".join(LABELS[s] for s in SYSTEMS) + " |"
    sep = "|---|" + "---|" * len(SYSTEMS)

    def row(label, cells):
        return f"| {label} | " + " | ".join(cells) + " |"

    def doc_cov(s):
        its = items(scores, s, "doc")
        found = sum(i["found"] for i in its)
        total = sum(i["total"] for i in its)
        return f"{pct(found, total)} ({found}/{total})"

    def doc_all(s):
        its = items(scores, s, "doc")
        return ratio(sum(1 for i in its if i["all_found"]), len(its))

    def abstained(qtype):
        def fn(s):
            its = items(scores, s, qtype)
            return ratio(sum(1 for i in its if i["abstained"]), len(its))
        return fn

    def versus(s):
        if s == "multi":
            return "n/a"
        b, t, w = paired(scores, "multi", s)
        return f"{b} / {t} / {w}"

    multi_all = [v["multi"] for v in scores.values() if "multi" in v]
    multi_web = items(scores, "multi", "web")
    na = ["n/a", "n/a"]
    review_entries = load_entries()
    extra_hits, review_problems = apply_review(results, scores, review_entries)

    def doc_cov_reviewed(s):
        its = items(scores, s, "doc")
        found = sum(i["found"] for i in its) + extra_hits.get(s, 0)
        total = sum(i["total"] for i in its)
        return f"{pct(found, total)} ({found}/{total})"

    lines = [
        "# Comparison results: baseline vs single call vs multi-agent",
        "",
        f"Run started {run['started']}. Model `{fp['model']}` via `{fp['provider']}`, extra request fields {fp['extra_body']}. "
        f"Retrieval: {fp['chunk_size']}-word chunks, overlap {fp['chunk_overlap']}, top-{fp['top_k']}, "
        f"embedding model `{fp['embedding_model']}`. Temperature 0 for every call. "
        f"Complete results: {n_complete} of {len(SYSTEMS) * len(QUESTIONS)}; errors: {n_error}.",
        "",
        "Systems: **baseline** = the M2 pipeline (one call, its original prompt, which says to answer only from the context and to say so when it is missing); "
        "**single call, writer prompt** = the same retrieval and one call using the multi-agent writer prompt, with no routing, review or web search "
        "(separates the effect of orchestration from the effect of prompt wording); **multi-agent** = supervisor routing, researcher (document retrieval "
        "plus web search when routed), writer, and supervisor review with at most one revision.",
        "",
        "Scoring is rule-based (`evaluation/scoring.py`). A fact counts only if the answer contains the expected wording after normalizing Unicode dashes, "
        "spaces and thousands separators; correct paraphrases count as misses for every system. Web questions are scored on routing and citation presence "
        "only: the correctness of web answers is not measured here. Raw answers are in `evaluation/results/run_main.json`.",
        "",
        "## Answer quality",
        "",
        head,
        sep,
        row("Answerable questions: key-fact coverage", [doc_cov(s) for s in SYSTEMS]),
        row("Answerable questions: coverage after paraphrase review (see the Paraphrase review section)", [doc_cov_reviewed(s) for s in SYSTEMS]),
        row("Answerable questions: all key facts found", [doc_all(s) for s in SYSTEMS]),
        row("Answerable questions: multi-agent found more / same / fewer facts than this system (questions)", [versus(s) for s in SYSTEMS]),
        row("Not-in-paper questions: abstained", [abstained("absent")(s) for s in SYSTEMS]),
        row("Web questions: said it could not answer", [abstained("web")(s) for s in SYSTEMS]),
        row("Web questions: routed to web search", na + [ratio(sum(1 for i in multi_web if i["routed_web"]), len(multi_web))]),
        row("Web questions: web results retrieved", na + [ratio(sum(1 for i in multi_web if i["web_results_retrieved"]), len(multi_web))]),
        row("Web questions: answer text contains a URL", na + [ratio(sum(1 for i in multi_web if i["url_in_answer"]), len(multi_web))]),
        row("Routing correct (all questions)", na + [ratio(sum(1 for i in multi_all if i["route_correct"]), len(multi_all))]),
        row("Reviewer requested a revision", na + [ratio(sum(1 for i in multi_all if i["revised"]), len(multi_all))]),
    ]

    for title, types in (("questions that need no web search", ("doc", "absent")), ("questions that need web search", ("web",))):
        stats = {s: cost_means(results, s, types) for s in SYSTEMS}
        lines += ["", f"## Cost per question (mean): {title}", "", head, sep,
                  row("Questions measured", [str(stats[s][0]) for s in SYSTEMS])]
        for field in COST_FIELDS:
            digits = 0 if field.endswith("tokens") else 1
            lines.append(row(field, [fmt(stats[s][1][field], digits) for s in SYSTEMS]))

    lines += [
        "",
        "`api_seconds` is time spent inside provider calls; `throttle_seconds` is time spent waiting to stay under the free-tier token limit; "
        "`wall_seconds` is both plus pipeline overhead. Compare systems on `api_seconds`, since waiting time depends on what ran just before.",
        "",
        "## Per-question detail",
        "",
        "| id | type | baseline | single call | multi-agent | route | revised | multi calls | multi prompt tokens |",
        "|---|---|---|---|---|---|---|---|---|",
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
            f"| {qid} | {q['type']} | {cell(qid, 'baseline')} | {cell(qid, 'single_writer')} | {cell(qid, 'multi')} | "
            f"{multi.get('route', 'n/a')} | {'yes' if multi.get('revisions', 0) >= 1 else 'no'} | "
            f"{multi.get('calls', 'n/a')} | {multi.get('prompt_tokens', 'n/a')} |"
        )
    if review_entries:
        lines += [
            "",
            "## Paraphrase review",
            "",
            "Strict scoring counts a correct paraphrase as a miss. The entries below are strict misses whose saved answer states the fact in different words; "
            "each quote is verified against the saved answer by the report script. The entries were proposed by the project's AI assistant and are to be confirmed by the author.",
            "",
            "| id | system | fact | supporting quote |",
            "|---|---|---|---|",
        ]
        for e in review_entries:
            fact = QMAP[e["id"]]["key_facts"][e["fact"]]["fact"]
            lines.append(f"| {e['id']} | {LABELS[e['system']]} | {fact} | {e['quote']} |")
        if review_problems:
            lines += ["", "REVIEW PROBLEMS (entries not counted):"] + [f"- {p}" for p in review_problems]
    return "\n".join(lines) + "\n"


def audit(results, scores, full=False):
    """Print raw answers for anything that was not a full pass, so scoring can be checked by eye."""
    for qid, entry in results.items():
        q = QMAP[qid]
        limit = None if (full and q["type"] != "absent") else 350
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
            text = " ".join(r["answer"].split())
            print(f"[{qid}] {system} {tag}\n    {text if limit is None else text[:limit]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--md", default=DEFAULT_MD)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-full", action="store_true", help="print complete raw answers (long)")
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
    if args.audit or args.audit_full:
        print("--- AUDIT ---")
        audit(results, scores, full=args.audit_full)


if __name__ == "__main__":
    main()
