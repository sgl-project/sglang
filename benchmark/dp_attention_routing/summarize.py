"""Turn driver.py results into the markdown tables in the PR.

Usage:
  python summarize.py RESULTS_DIR [--baseline main] [--candidate fix]

RESULTS_DIR holds one subdirectory per server run, named LABEL as passed to
run_bench.sh. A label is BUILD[digits][-CONFIG], e.g. main1, fix2, main-mixed,
fix-mixsmall. Runs with the same BUILD and CONFIG are pooled; a label without
-CONFIG belongs to config "default".

Metric definitions (short = the 1024-token requests, long = 32K/64K prompts):
  * busier-rank share, output tok/s, long TTFT, inter-token p99, >500 ms gaps:
    the per-trial values from driver.py, median over trials.
  * short TTFT and short end-to-end latency (first send to last token):
    pooled over every short request of every trial; mean, and p90 by the
    nearest-rank method.
  * worst stall: each short request's longest inter-token gap, pooled; p90/p99.
  * decode phase: first to last token of a short request, pooled mean.
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROFILES = ("burst", "staggered")
SHORT_PROMPT = 1024
LABEL = re.compile(r"^(?P<build>[A-Za-z_]+?)\d*(?:-(?P<config>.+))?$")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_trials(results_dir):
    """{(build, config): {profile: [trial, ...]}}"""
    groups = defaultdict(lambda: defaultdict(list))
    for run in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        match = LABEL.match(run.name)
        if not match:
            continue
        key = (match["build"], match["config"] or "default")
        for profile in PROFILES:
            for path in sorted(run.glob(f"{profile}-*.json")):
                groups[key][profile].append(json.loads(path.read_text()))
    return groups


def metrics(trials):
    summaries = [t["summary"] for t in trials]
    shorts = [
        r for t in trials for r in t["requests"] if r["prompt_tokens"] == SHORT_PROMPT
    ]
    ttft = [r["ttft_ms"] / 1000 for r in shorts]
    e2e = [r["events"][-1][0] - r["sent_s"] for r in shorts]
    worst = [
        max(b[0] - a[0] for a, b in zip(r["events"], r["events"][1:])) for r in shorts
    ]
    decode = [r["events"][-1][0] - r["events"][0][0] for r in shorts]

    def median(key):
        return statistics.median(s[key] for s in summaries)

    return {
        "trials": len(trials),
        "share": median("hot_rank_prompt_share_pct"),
        "tok_s": median("output_tok_s"),
        "ttft_mean": statistics.mean(ttft),
        "ttft_p90": percentile(ttft, 0.9),
        "e2e_mean": statistics.mean(e2e),
        "long32": statistics.median(s["long_ttft_ms"]["32768"] for s in summaries)
        / 1000,
        "long64": statistics.median(s["long_ttft_ms"]["65536"] for s in summaries)
        / 1000,
        "worst_p90": percentile(worst, 0.9),
        "worst_p99": percentile(worst, 0.99),
        "gap_p99_ms": median("short_gap_p99_ms"),
        "gaps_500": median("short_gaps_over_500ms_per_request"),
        "decode_mean": statistics.mean(decode),
    }


def pct_change(new, old):
    return f"{100 * (new / old - 1):+.1f}%"


def print_config(config, base, cand, base_name, cand_name):
    print(
        f"\n### config: {config}  ({base['burst']['trials']} trials per profile per build)\n"
    )
    print(f"| profile | metric | {base_name} | {cand_name} | change |")
    print("|---|---|---:|---:|---:|")
    for profile in PROFILES:
        b, c = base[profile], cand[profile]
        rows = [
            (
                "busier rank's share of input tokens (median)",
                f"{b['share']:.1f}%",
                f"{c['share']:.1f}%",
                "",
            ),
            (
                "output throughput (median)",
                f"{b['tok_s']:.1f} tok/s",
                f"{c['tok_s']:.1f} tok/s",
                pct_change(c["tok_s"], b["tok_s"]),
            ),
            (
                "short-request TTFT, mean / p90",
                f"{b['ttft_mean']:.2f} / {b['ttft_p90']:.2f} s",
                f"{c['ttft_mean']:.2f} / {c['ttft_p90']:.2f} s",
                f"{pct_change(c['ttft_mean'], b['ttft_mean'])} / {pct_change(c['ttft_p90'], b['ttft_p90'])}",
            ),
            (
                "short-request end-to-end latency, mean",
                f"{b['e2e_mean']:.2f} s",
                f"{c['e2e_mean']:.2f} s",
                pct_change(c["e2e_mean"], b["e2e_mean"]),
            ),
            (
                "long-prompt TTFT 32K / 64K (median)",
                f"{b['long32']:.2f} / {b['long64']:.2f} s",
                f"{c['long32']:.2f} / {c['long64']:.2f} s",
                "",
            ),
            (
                "worst stall per short request, p90 / p99",
                f"{b['worst_p90']:.2f} / {b['worst_p99']:.2f} s",
                f"{c['worst_p90']:.2f} / {c['worst_p99']:.2f} s",
                "",
            ),
            (
                "inter-token p99 (median)",
                f"{b['gap_p99_ms']:.0f} ms",
                f"{c['gap_p99_ms']:.0f} ms",
                pct_change(c["gap_p99_ms"], b["gap_p99_ms"]),
            ),
            (
                ">500 ms gaps per short request (median)",
                f"{b['gaps_500']:.2f}",
                f"{c['gaps_500']:.2f}",
                "",
            ),
            (
                "decode phase per short request, mean",
                f"{b['decode_mean']:.2f} s",
                f"{c['decode_mean']:.2f} s",
                "",
            ),
        ]
        for name, bv, cv, change in rows:
            print(f"| {profile} | {name} | {bv} | {cv} | {change} |")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results_dir")
    parser.add_argument("--baseline", default="main")
    parser.add_argument("--candidate", default="fix")
    args = parser.parse_args()
    groups = load_trials(args.results_dir)
    configs = sorted(
        {config for _, config in groups}, key=lambda c: (c != "default", c)
    )
    for config in configs:
        base = groups.get((args.baseline, config))
        cand = groups.get((args.candidate, config))
        if not base or not cand:
            print(
                f"\n(config {config}: missing {args.baseline} or {args.candidate} runs, skipped)"
            )
            continue
        print_config(
            config,
            {p: metrics(base[p]) for p in PROFILES},
            {p: metrics(cand[p]) for p in PROFILES},
            args.baseline,
            args.candidate,
        )


if __name__ == "__main__":
    main()
