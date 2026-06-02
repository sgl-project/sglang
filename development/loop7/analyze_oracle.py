"""Loop-7 M0 oracle aggregator — turn the per-(trial,layer) sink records into the
budget-vs-scorer artifact, per length, with the oracle-uplift gate.

Reads the JSONL sink written by the server's recall oracle, groups SUCCESS
records by length (parsed from request_id "L<len>-i<idx>"), and computes
score-only recall@K plus needle-rank stats. FAILURE markers (no_active_trial /
span_out_of_range / exception) are counted and reported — a length with zero
success records is flagged (the old silent-64K regression).

Verdict per length:
  budget-limited  — score-only recall@4096 (a feasible budget) materially
                    exceeds recall@2048 (the locked index_topk); a wider-budget
                    decode recovers the needle.
  scorer-limited  — no feasible budget (<=8192) recovers it; only a better
                    selector would.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
from collections import defaultdict

K_GRID = (512, 1024, 2048, 4096, 8192)
UPLIFT_EPS = 0.10  # recall@4096 must beat recall@2048 by >10pp to call budget-limited


def _length_of(request_id: str):
    # "L16384-i3" -> 16384
    try:
        return int(request_id.split("-")[0].lstrip("L"))
    except (ValueError, AttributeError):
        return None


def _recall_bool(rec, k):
    rk = rec.get("recall_at_k", {})
    # JSON stringifies int dict keys; accept both.
    if str(k) in rk:
        return bool(rk[str(k)])
    if k in rk:
        return bool(rk[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sink", default=os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH"))
    ap.add_argument("--out", default="development/loop7/oracle_budget_vs_scorer_r4.json")
    ap.add_argument(
        "--op-point",
        default="DS int8 / mem 0.7 / TP=8, eager (--disable-cuda-graph), recall_oracle config-borne",
    )
    args = ap.parse_args()

    sink_path = args.sink
    if not sink_path:
        from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as s

        sink_path = s.default_sink_path()
    if not os.path.exists(sink_path):
        raise SystemExit(f"sink not found: {sink_path}")

    by_len_success = defaultdict(list)
    failures = defaultdict(int)
    with open(sink_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "failure" in rec:
                failures[str(rec["failure"]).split(":")[0]] += 1
                continue
            L = _length_of(rec.get("request_id", ""))
            if L is not None:
                by_len_success[L].append(rec)

    out = {
        "op_point": args.op_point,
        "method": (
            "score-only recall@K on live all-reduced score tensor (after all_reduce, "
            "before top-K); needle span via raw-prompt offset mapping; per (trial,layer) "
            "sample; recall@K = all needle tokens ranked < K (needle_worst_rank < K)"
        ),
        "index_topk": 2048,
        "sink": sink_path,
        "failure_markers": dict(failures),
        "lengths": {},
    }

    for L in sorted(by_len_success):
        recs = by_len_success[L]
        trials = sorted({r.get("request_id") for r in recs})
        recall = {}
        for k in K_GRID:
            vals = [_recall_bool(r, k) for r in recs]
            vals = [v for v in vals if v is not None]
            recall[k] = (sum(vals) / len(vals)) if vals else None
        ranks = [int(r["needle_worst_rank"]) for r in recs if "needle_worst_rank" in r]
        r2048 = recall.get(2048) or 0.0
        r4096 = recall.get(4096) or 0.0
        r8192 = recall.get(8192) or 0.0
        budget_limited = (max(r4096, r8192) - r2048) > UPLIFT_EPS
        out["lengths"][str(L)] = {
            "trials": len(trials),
            "layer_samples": len(recs),
            "score_only_recall_at_k": {str(k): recall[k] for k in K_GRID},
            "needle_rank_min": min(ranks) if ranks else None,
            "needle_rank_median": int(stats.median(ranks)) if ranks else None,
            "needle_rank_max": max(ranks) if ranks else None,
            "uplift_gate_recall4096_minus_recall2048": round(max(r4096, r8192) - r2048, 4),
            "verdict": "budget-limited" if budget_limited else "scorer-limited",
        }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote -> {args.out}")
    if failures:
        print(f"failure markers: {dict(failures)}")


if __name__ == "__main__":
    main()
