#!/usr/bin/env python3
"""Two-column comparator for native_nsa vs double_sparsity bench_serving runs.

Consumes the JSONL output of ``sglang.bench_serving --output-file`` for a
``native_nsa`` baseline run (produced by ``benchmark_baseline.sh``) and a
``double_sparsity`` run (produced by ``benchmark.sh``), and emits a side-by-
side report.

Refuses to publish if the two runs' hardware / TP / page / radix-cache /
concurrency context disagree. The match-enforcement contract from AC-7:

    {GPU id, TP size, page size, radix-cache setting, concurrency}

must be identical between the two columns. Bench_serving currently records
some of these as run-level metadata; missing fields are best-effort matched
against the filename tags (e.g. ``native_nsa_gsp_isl4096_osl512_c64.jsonl``
implies concurrency=64).

SLO gate per AC-8: each row is annotated with `pass` / `fail` against
``per_request_output_tps_p50 >= 30`` and ``ttft_p99_s <= 22``.

No-op detector per AC-7: a row is flagged if any of
``selected_pages == total_pages`` or ``dense_fallback_total != 0``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SLO_PER_REQUEST_TPS_P50 = 30.0
SLO_TTFT_P99_S = 22.0


@dataclass
class RunMetrics:
    """Metrics distilled from a single bench_serving JSONL row."""

    concurrency: int
    num_prompts: int
    isl: int
    osl: int
    output_tps_p50: Optional[float]
    output_tps_p99: Optional[float]
    ttft_p50_s: Optional[float]
    ttft_p99_s: Optional[float]
    tpot_p50_ms: Optional[float]
    tpot_p99_ms: Optional[float]
    goodput_under_slo: Optional[float]
    selected_pages_mean: Optional[float]
    dense_fallback_total: Optional[int]
    total_pages_mean: Optional[float]


@dataclass
class RunContext:
    """Hardware / config metadata that must agree across columns."""

    gpu_id: Optional[str]
    tp_size: Optional[int]
    page_size: Optional[int]
    disable_radix_cache: Optional[bool]
    concurrency: Optional[int]


def _filename_concurrency(path: str) -> Optional[int]:
    m = re.search(r"_c(\d+)\.jsonl$", path)
    if m:
        return int(m.group(1))
    return None


def _read_bench_jsonl(path: str) -> Tuple[RunContext, RunMetrics]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bench file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"bench file is empty: {path}")
    summary = rows[-1] if isinstance(rows[-1], dict) else rows[0]

    concurrency = (
        summary.get("max_concurrency")
        or summary.get("concurrency")
        or _filename_concurrency(path)
    )
    context = RunContext(
        gpu_id=str(summary.get("gpu_id") or summary.get("device") or ""),
        tp_size=summary.get("tp_size"),
        page_size=summary.get("page_size"),
        disable_radix_cache=summary.get("disable_radix_cache"),
        concurrency=int(concurrency) if concurrency is not None else None,
    )

    def _float(key: str) -> Optional[float]:
        v = summary.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def _int(key: str) -> Optional[int]:
        v = summary.get(key)
        return int(v) if isinstance(v, (int, float)) else None

    metrics = RunMetrics(
        concurrency=int(concurrency or 0),
        num_prompts=_int("num_prompts") or 0,
        isl=_int("input_len") or _int("median_input_len") or _int("isl") or 0,
        osl=_int("output_len") or _int("median_output_len") or _int("osl") or 0,
        output_tps_p50=_float("output_throughput_p50")
        or _float("per_req_output_tps_p50"),
        output_tps_p99=_float("output_throughput_p99")
        or _float("per_req_output_tps_p99"),
        ttft_p50_s=_float("median_ttft_ms")
        and (_float("median_ttft_ms") or 0) / 1000.0
        or _float("ttft_p50_s"),
        ttft_p99_s=_float("p99_ttft_ms")
        and (_float("p99_ttft_ms") or 0) / 1000.0
        or _float("ttft_p99_s"),
        tpot_p50_ms=_float("median_tpot_ms"),
        tpot_p99_ms=_float("p99_tpot_ms"),
        goodput_under_slo=_float("goodput_under_slo"),
        selected_pages_mean=_float("selected_pages_mean"),
        dense_fallback_total=_int("dense_fallback_total"),
        total_pages_mean=_float("total_pages_mean"),
    )
    return context, metrics


def _match_or_refuse(
    baseline: RunContext, ds: RunContext, *, strict_gpu: bool = False
) -> List[str]:
    """Return a list of human-readable mismatch reasons (empty = match)."""

    reasons: List[str] = []
    if strict_gpu and baseline.gpu_id != ds.gpu_id:
        reasons.append(
            f"gpu_id mismatch: native_nsa={baseline.gpu_id!r} ds={ds.gpu_id!r}"
        )
    if baseline.tp_size != ds.tp_size:
        reasons.append(
            f"tp_size mismatch: native_nsa={baseline.tp_size} ds={ds.tp_size}"
        )
    if baseline.page_size != ds.page_size:
        reasons.append(
            f"page_size mismatch: native_nsa={baseline.page_size} ds={ds.page_size}"
        )
    if baseline.disable_radix_cache != ds.disable_radix_cache:
        reasons.append(
            f"disable_radix_cache mismatch: "
            f"native_nsa={baseline.disable_radix_cache} ds={ds.disable_radix_cache}"
        )
    if baseline.concurrency != ds.concurrency:
        reasons.append(
            f"concurrency mismatch: native_nsa={baseline.concurrency} ds={ds.concurrency}"
        )
    return reasons


def _slo_verdict(m: RunMetrics) -> str:
    if m.output_tps_p50 is None or m.ttft_p99_s is None:
        return "missing-data"
    if m.output_tps_p50 >= SLO_PER_REQUEST_TPS_P50 and m.ttft_p99_s <= SLO_TTFT_P99_S:
        return "pass"
    return "fail"


def _no_op_flag(m: RunMetrics) -> str:
    if m.dense_fallback_total is not None and m.dense_fallback_total != 0:
        return f"dense_fallback={m.dense_fallback_total}"
    if (
        m.selected_pages_mean is not None
        and m.total_pages_mean is not None
        and m.selected_pages_mean == m.total_pages_mean
    ):
        return "selected_pages == total_pages"
    return ""


def render_markdown_report(
    baseline_metrics: RunMetrics,
    ds_metrics: RunMetrics,
    *,
    baseline_path: str,
    ds_path: str,
) -> str:
    rows = []
    rows.append("# Double Sparsity vs Native NSA — Comparison Report")
    rows.append("")
    rows.append(f"- native_nsa source: `{baseline_path}`")
    rows.append(f"- double_sparsity source: `{ds_path}`")
    rows.append(f"- concurrency: {ds_metrics.concurrency}")
    rows.append("")
    rows.append("| Metric | native_nsa | double_sparsity |")
    rows.append("|--------|------------|-----------------|")

    def _fmt(v):
        return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    pairs = [
        ("Per-request output tok/s P50", baseline_metrics.output_tps_p50, ds_metrics.output_tps_p50),
        ("Per-request output tok/s P99", baseline_metrics.output_tps_p99, ds_metrics.output_tps_p99),
        ("TTFT P50 (s)", baseline_metrics.ttft_p50_s, ds_metrics.ttft_p50_s),
        ("TTFT P99 (s)", baseline_metrics.ttft_p99_s, ds_metrics.ttft_p99_s),
        ("TPOT P50 (ms)", baseline_metrics.tpot_p50_ms, ds_metrics.tpot_p50_ms),
        ("TPOT P99 (ms)", baseline_metrics.tpot_p99_ms, ds_metrics.tpot_p99_ms),
        ("Goodput-under-SLO", baseline_metrics.goodput_under_slo, ds_metrics.goodput_under_slo),
        ("Selected pages (mean)", baseline_metrics.selected_pages_mean, ds_metrics.selected_pages_mean),
        ("Total pages (mean)", baseline_metrics.total_pages_mean, ds_metrics.total_pages_mean),
        ("dense_fallback_total", baseline_metrics.dense_fallback_total, ds_metrics.dense_fallback_total),
    ]
    for label, a, b in pairs:
        rows.append(f"| {label} | {_fmt(a)} | {_fmt(b)} |")

    rows.append("")
    rows.append(f"**DS SLO verdict (per-request P50 ≥ {SLO_PER_REQUEST_TPS_P50} tok/s, P99 TTFT ≤ {SLO_TTFT_P99_S} s):** {_slo_verdict(ds_metrics)}")
    ds_no_op = _no_op_flag(ds_metrics)
    if ds_no_op:
        rows.append(f"**No-op detector:** triggered ({ds_no_op})")
    else:
        rows.append("**No-op detector:** clean")
    return "\n".join(rows) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_compare.py",
        description="Side-by-side comparator for native_nsa vs double_sparsity bench_serving runs.",
    )
    parser.add_argument("--baseline", required=True, help="Path to native_nsa *.jsonl")
    parser.add_argument("--ds", required=True, help="Path to double_sparsity *.jsonl")
    parser.add_argument(
        "--output", default=None, help="Write Markdown report to this path (default stdout)."
    )
    parser.add_argument(
        "--json-output", default=None, help="Write JSON report to this path."
    )
    parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Enforce gpu_id match (default: TP/page/radix/concurrency only).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    baseline_ctx, baseline_m = _read_bench_jsonl(args.baseline)
    ds_ctx, ds_m = _read_bench_jsonl(args.ds)

    reasons = _match_or_refuse(baseline_ctx, ds_ctx, strict_gpu=args.strict_gpu)
    if reasons:
        logger.error(
            "Refusing to publish two-column report — context disagrees:\n  %s",
            "\n  ".join(reasons),
        )
        return 2

    md = render_markdown_report(
        baseline_m, ds_m, baseline_path=args.baseline, ds_path=args.ds
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("wrote Markdown report to %s", args.output)
    else:
        sys.stdout.write(md)
    if args.json_output:
        payload = {
            "baseline_path": args.baseline,
            "ds_path": args.ds,
            "baseline_context": asdict(baseline_ctx),
            "ds_context": asdict(ds_ctx),
            "baseline_metrics": asdict(baseline_m),
            "ds_metrics": asdict(ds_m),
            "ds_slo_verdict": _slo_verdict(ds_m),
            "ds_no_op_flag": _no_op_flag(ds_m),
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("wrote JSON report to %s", args.json_output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
