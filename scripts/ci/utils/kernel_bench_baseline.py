#!/usr/bin/env python3
"""Build a per-GPU baseline from JIT kernel benchmark records and compare a run with it.

Records are the JSONL lines the benchmark marker writes under
``SGLANG_JIT_BENCHMARK_OUTPUT`` (``sglang/kernels/jit/benchmark/marker/record.py``).
Standard library only, so it runs on CPU runners without installing sglang.

    kernel_bench_baseline.py build --inputs DIR... --out baseline.json [--window 7]
    kernel_bench_baseline.py compare --baseline baseline.json --current DIR...
        [--out-json F] [--out-md F]

A case is keyed by ``(case_id, env.gpu_name)``. ``compare`` checks only the cases
present in both the baseline and the current run; anything else is ignored.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

RECORD_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1
# Run-to-run noise allowance on top of the record's tolerance, in baseline MADs.
NOISE_MADS = 3.0

Key = Tuple[str, str]


def _paths(inputs: Iterable[str]) -> List[str]:
    out: List[str] = []
    for path in inputs:
        if os.path.isdir(path):
            out += sorted(
                glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
            )
        elif os.path.isfile(path):
            out.append(path)
        else:
            print(f"skipping missing input {path}", file=sys.stderr)
    return out


def load_records(inputs: Iterable[str]) -> List[Dict[str, Any]]:
    """Records of the supported schema (`ok` and `skip`); others are dropped."""
    records = []
    for path in _paths(inputs):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("schema_version") == RECORD_SCHEMA_VERSION:
                    records.append(rec)
    return records


def _ok(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r["status"] == "ok"]


def _key(rec: Dict[str, Any]) -> Key:
    return rec["case_id"], rec["env"]["gpu_name"]


def _run_of(rec: Dict[str, Any]) -> str:
    return rec["run"]["id"] or rec["run"]["process"]


def build(records: List[Dict[str, Any]], *, window: int) -> Dict[str, Any]:
    """Median p50 and MAD of each case over its latest `window` runs."""
    by_key: Dict[Key, Dict[str, Dict[str, Any]]] = {}
    for rec in _ok(records):
        runs = by_key.setdefault(_key(rec), {})
        run = _run_of(rec)
        # A retried case appears twice in one run; keep the later measurement.
        if run not in runs or rec["timestamp"] > runs[run]["timestamp"]:
            runs[run] = rec
    cases = []
    for (case_id, gpu), runs in sorted(by_key.items()):
        latest = sorted(runs.values(), key=lambda r: r["timestamp"])[-window:]
        p50s = [r["metrics"]["p50_us"] for r in latest]
        median = statistics.median(p50s)
        last = latest[-1]
        cases.append(
            {
                "case_id": case_id,
                "gpu_name": gpu,
                "bench_id": last["bench_id"],
                "params": last["params"],
                "line": last["line"],
                "p50_us": median,
                "mad_us": statistics.median(abs(x - median) for x in p50s),
                "n_runs": len(p50s),
                "commit": last["env"]["commit"],
            }
        )
    return {"schema_version": BASELINE_SCHEMA_VERSION, "cases": cases}


def _threshold_us(
    *, tolerance: Dict[str, float], base_us: float, mad_us: float
) -> float:
    return max(tolerance["rel"] * base_us, tolerance["abs_us"], NOISE_MADS * mad_us)


def compare(
    baseline: Dict[str, Any], records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """One row per case in both sets. The gate and tolerance come from the
    current record, so a change that edits a benchmark's annotation applies it."""
    base = {(c["case_id"], c["gpu_name"]): c for c in baseline["cases"]}
    current: Dict[Key, List[Dict[str, Any]]] = {}
    for rec in _ok(records):
        if _key(rec) in base:
            current.setdefault(_key(rec), []).append(rec)
    rows = []
    for key, recs in sorted(current.items()):
        last = max(recs, key=lambda r: r["timestamp"])
        b = base[key]
        cur_us = statistics.median(r["metrics"]["p50_us"] for r in recs)
        delta_us = cur_us - b["p50_us"]
        threshold_us = _threshold_us(
            tolerance=last["tolerance"], base_us=b["p50_us"], mad_us=b["mad_us"]
        )
        if delta_us > threshold_us:
            verdict = "REGRESS"
        elif delta_us < -threshold_us:
            verdict = "IMPROVE"
        else:
            verdict = "PASS"
        rows.append(
            {
                "case_id": key[0],
                "gpu_name": key[1],
                "bench_id": last["bench_id"],
                "params": last["params"],
                "line": last["line"],
                "gate": last["gate"],
                "base_us": b["p50_us"],
                "cur_us": cur_us,
                "delta_rel": delta_us / b["p50_us"] if b["p50_us"] > 0 else 0.0,
                "threshold_us": threshold_us,
                "base_runs": b["n_runs"],
                "verdict": verdict,
            }
        )
    return rows


def render_md(
    rows: List[Dict[str, Any]], *, skipped: int = 0, max_rows: int = 60
) -> str:
    gated = [r for r in rows if r["gate"]]
    flagged = sorted(
        (r for r in gated if r["verdict"] != "PASS"),
        key=lambda r: -abs(r["delta_rel"]),
    )
    regress = sum(r["verdict"] == "REGRESS" for r in gated)
    improve = sum(r["verdict"] == "IMPROVE" for r in gated)
    lines = ["## JIT kernel benchmark vs. nightly baseline", ""]
    skip_note = f" {skipped} cases were skipped in this run." if skipped else ""
    if not rows:
        lines.append(
            "No case of this run has a baseline counterpart (same case_id and GPU)."
            + skip_note
        )
        return "\n".join(lines) + "\n"
    gpus = ", ".join(sorted({r["gpu_name"] for r in rows}))
    lines.append(
        f"{len(rows)} cases compared on {gpus} ({len(gated)} gated): "
        f"**{regress} slower**, {improve} faster beyond tolerance.{skip_note}"
    )
    if flagged:
        lines += [
            "",
            "| verdict | benchmark | params | line | GPU | base us | this run us | delta | threshold us | baseline runs |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in flagged[:max_rows]:
            params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            lines.append(
                f"| {r['verdict']} | `{r['bench_id'].rsplit('/', 1)[-1]}` | {params} "
                f"| {r['line']} | {r['gpu_name']} | {r['base_us']:.3f} | {r['cur_us']:.3f} "
                f"| {r['delta_rel']:+.1%} | {r['threshold_us']:.3f} | {r['base_runs']} |"
            )
        if len(flagged) > max_rows:
            lines.append(f"\n... and {len(flagged) - max_rows} more (see the JSON).")
    return "\n".join(lines) + "\n"


def _write(path: Optional[str], text: str) -> None:
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="baseline from records of recent runs")
    p_build.add_argument("--inputs", nargs="+", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--window", type=int, default=7, help="latest runs per case")
    p_cmp = sub.add_parser("compare", help="compare a run with a baseline")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--current", nargs="+", required=True)
    p_cmp.add_argument("--out-json")
    p_cmp.add_argument("--out-md")
    p_cmp.add_argument(
        "--fail-on-regress",
        action="store_true",
        help="exit 1 (after writing the reports) when a gated case regressed",
    )
    args = parser.parse_args()

    if args.cmd == "build":
        baseline = build(load_records(args.inputs), window=args.window)
        _write(args.out, json.dumps(baseline, indent=1))
        print(f"baseline: {len(baseline['cases'])} cases -> {args.out}")
        return

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    if baseline.get("schema_version") != BASELINE_SCHEMA_VERSION:
        md = "## JIT kernel benchmark vs. nightly baseline\n\nBaseline schema differs from this run's tool; comparison skipped.\n"
        rows: List[Dict[str, Any]] = []
    else:
        records = load_records(args.current)
        rows = compare(baseline, records)
        md = render_md(rows, skipped=sum(r["status"] == "skip" for r in records))
    _write(args.out_json, json.dumps(rows, indent=1))
    _write(args.out_md, md)
    print(md)
    regressed = [r for r in rows if r["gate"] and r["verdict"] == "REGRESS"]
    if regressed and args.fail_on_regress:
        # GitHub Actions annotation, shown on the run and the PR checks page.
        print(
            f"::error title=JIT kernel benchmark regression::{len(regressed)} gated "
            "cases are slower than the nightly baseline; see the job summary."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
