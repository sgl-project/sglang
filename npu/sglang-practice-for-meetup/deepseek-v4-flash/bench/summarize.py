#!/usr/bin/env python3
"""
summarize.py — turn benchmark output into the measurement tables.

Inputs (written by bench/run_matrix.sh):
  results/<cell>.jsonl      SGLang `bench_serving --output-file` output (one JSON object per run,
                            with per-request arrays when --output-details is used)
  results/<cell>.meta.json  wall-clock + command + hardware metadata for that cell

Outputs:
  results/summary.json      machine-readable, one object per cell
  docs/measurements.md      the auto-generated tables between the AUTO markers (human notes are preserved)

Goodput is computed here rather than taken from the engine: a request counts only if it satisfies both
SLOs, which is what a serving SLO actually means. Defaults: TTFT <= 2000 ms, TPOT <= 100 ms.
Override with --slo-ttft-ms / --slo-tpot-ms.

Usage: python3 bench/summarize.py [--results results] [--out docs/measurements.md]
"""
from __future__ import annotations

import argparse
import json
import glob
import os
import statistics
from datetime import datetime, timezone


def pct(values, q):
    """Nearest-rank percentile; q in (0,100]."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = max(1, min(len(s), int(round((q / 100.0) * len(s) + 0.5)) - 1))
    return s[k]


def fnum(x, nd=1):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def load_cell(jsonl_path):
    """Return the last complete JSON object in the jsonl file."""
    if not os.path.exists(jsonl_path):
        return None
    last = None
    with open(jsonl_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                last = obj
    return last


def goodput_from_details(obj, slo_ttft, slo_tpot, duration_s):
    """Compute SLO-attainment from the per-request arrays, if present."""
    ttfts = obj.get("ttfts")
    itls = obj.get("itls")
    if not ttfts or not isinstance(ttfts, list):
        return None, None, None
    good = 0
    tpot_values = []
    for i, ttft in enumerate(ttfts):
        tpot = None
        if isinstance(itls, list) and i < len(itls) and isinstance(itls[i], list) and len(itls[i]) > 1:
            tpot = statistics.mean(itls[i][1:])
        elif isinstance(itls, list) and i < len(itls) and isinstance(itls[i], (int, float)):
            tpot = float(itls[i])
        if tpot is not None:
            tpot_values.append(tpot)
        ok = (ttft is not None and ttft <= slo_ttft) and (tpot is None or tpot <= slo_tpot)
        if ok:
            good += 1
    n = len(ttfts)
    attainment = good / n if n else None
    gp = (good / duration_s) if duration_s else None
    med_tpot = statistics.median(tpot_values) if tpot_values else None
    return attainment, gp, med_tpot


def collect(results_dir, slo_ttft, slo_tpot):
    cells = []
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "*.meta.json"))):
        cell = os.path.basename(meta_path)[: -len(".meta.json")]
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
        jsonl = os.path.join(results_dir, f"{cell}.jsonl")
        obj = load_cell(jsonl) or {}
        duration = meta.get("duration_s") or obj.get("duration_s")

        attainment, goodput, med_tpot_from_details = goodput_from_details(
            obj, slo_ttft, slo_tpot, duration or 0
        )
        tpot_median = obj.get("median_tpot_ms") or med_tpot_from_details

        row = {
            "cell": cell,
            "input_len": meta.get("input_len", obj.get("random_input_len")),
            "output_len": meta.get("output_len", obj.get("random_output_len")),
            "num_prompts": meta.get("num_prompts", obj.get("num_prompts")),
            "max_concurrency": meta.get("max_concurrency"),
            "request_rate": meta.get("request_rate"),
            "duration_s": duration,
            "completed": obj.get("completed"),
            "request_throughput": obj.get("request_throughput"),
            "input_throughput": obj.get("input_throughput"),
            "output_throughput": obj.get("output_throughput"),
            "total_throughput": obj.get("total_throughput"),
            "ttft_mean_ms": obj.get("mean_ttft_ms"),
            "ttft_p50_ms": obj.get("median_ttft_ms"),
            "ttft_p90_ms": obj.get("p90_ttft_ms"),
            "ttft_p99_ms": obj.get("p99_ttft_ms"),
            "tpot_mean_ms": obj.get("mean_tpot_ms"),
            "tpot_p50_ms": tpot_median,
            "tpot_p90_ms": obj.get("p90_tpot_ms"),
            "tpot_p99_ms": obj.get("p99_tpot_ms"),
            "itl_mean_ms": obj.get("mean_itl_ms"),
            "itl_p99_ms": obj.get("p99_itl_ms"),
            "max_itl_ms": obj.get("max_itl_ms"),
            "e2e_p50_ms": obj.get("median_e2e_latency_ms"),
            "e2e_p99_ms": obj.get("p99_e2e_latency_ms"),
            "slo_attainment": attainment,
            "goodput_req_s": goodput,
            "return_code": meta.get("return_code"),
            "meta": meta,
        }
        cells.append(row)
    return cells


def render_markdown(cells, slo_ttft, slo_tpot):
    lines = []
    if not cells:
        return "_No results yet. Run `bench/run_matrix.sh` after starting the server._\n"

    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                 f"by `bench/summarize.py`. Goodput SLO: TTFT ≤ {slo_ttft:.0f} ms and TPOT ≤ {slo_tpot:.0f} ms._\n")

    m = cells[0]["meta"] or {}
    hw = m.get("hardware") or {}
    lines.append("| Configuration | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| hardware | chip `{hw.get('chip', '?')}`, {hw.get('npus', '?')} devices, "
                 f"tp={hw.get('tp', '?')}, dp={hw.get('dp', '?')} |")
    lines.append(f"| image | `{m.get('image', '?')}` |")
    lines.append(f"| model dir | `{m.get('model_dir', '?')}` |")
    lines.append("")

    lines.append("| cell | in/out | prompts | conc / rate | dur s | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p99 | SLO ok | goodput |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in cells:
        conc = c["max_concurrency"] if c["max_concurrency"] is not None else f"rate {c['request_rate']}"
        attainment = c["slo_attainment"]
        lines.append(
            "| {cell} | {i}/{o} | {n} | {conc} | {dur} | {rps} | {otps} | {ttft50} | {ttft99} | "
            "{tpot50} | {tpot99} | {e2e99} | {slo} | {gp} |".format(
                cell=c["cell"], i=c["input_len"], o=c["output_len"], n=c["num_prompts"],
                conc=conc, dur=fnum(c["duration_s"], 1),
                rps=fnum(c["request_throughput"], 3),
                otps=fnum(c["output_throughput"], 1),
                ttft50=fnum(c["ttft_p50_ms"], 1), ttft99=fnum(c["ttft_p99_ms"], 1),
                tpot50=fnum(c["tpot_p50_ms"], 2), tpot99=fnum(c["tpot_p99_ms"], 2),
                e2e99=fnum(c["e2e_p99_ms"], 1),
                slo="—" if attainment is None else f"{attainment * 100:.0f}%",
                gp=fnum(c["goodput_req_s"], 3),
            )
        )
    lines.append("")
    lines.append("Raw records: `results/<cell>.jsonl` (per-run summary + per-request arrays when "
                 "`--output-details` was used); metadata and the exact command: `results/<cell>.meta.json`.")
    return "\n".join(lines) + "\n"


def update_measurements(path, block):
    start = "<!-- AUTO:RESULTS:START -->"
    end = "<!-- AUTO:RESULTS:END -->"
    text = open(path).read() if os.path.exists(path) else f"# Measurements\n\n{start}\n{end}\n"
    if start not in text or end not in text:
        text = text.rstrip() + f"\n\n{start}\n{end}\n"
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    open(path, "w").write(pre + start + "\n" + block + "\n" + end + post)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(proj, "results"))
    ap.add_argument("--out", default=os.path.join(proj, "docs", "measurements.md"))
    ap.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=100.0)
    args = ap.parse_args()

    cells = collect(args.results, args.slo_ttft_ms, args.slo_tpot_ms)
    block = render_markdown(cells, args.slo_ttft_ms, args.slo_tpot_ms)
    print(block)
    summary_path = os.path.join(args.results, "summary.json")
    json.dump(cells, open(summary_path, "w"), indent=2, default=str)
    update_measurements(args.out, block)
    print(f"wrote {summary_path}")
    print(f"updated {args.out}")


if __name__ == "__main__":
    main()
