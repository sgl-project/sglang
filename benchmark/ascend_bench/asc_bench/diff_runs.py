"""Regression diff: join two run manifests by cell_hash and report deltas."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from asc_bench.runner import Manifest
from asc_bench.sla import aggregate_by_hash, parse_last_metrics

DEFAULT_METRICS = ["output_throughput", "p99_ttft_ms", "p99_tpot_ms"]


def _collect(run_dir: Path) -> tuple[dict[str, list[dict[str, float]]], dict[str, str]]:
    """Map cell_hash -> list of metric dicts, and hash -> representative id."""
    manifest = run_dir / "manifest.jsonl"
    per_cell: dict[str, dict[str, float]] = {}
    hash_of: dict[str, str] = {}
    for record in Manifest.result_rows(manifest):
        cell_id = record["cell_id"]
        metrics = record.get("metrics")
        if isinstance(metrics, dict) and metrics:
            per_cell[cell_id] = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
        else:
            metrics_path = run_dir / "cells" / cell_id / "bench.jsonl"
            parsed = parse_last_metrics(metrics_path)
            if parsed:
                per_cell[cell_id] = {
                    k: float(v)
                    for k, v in parsed.items()
                    if isinstance(v, (int, float))
                }
        if cell_id not in hash_of:
            hash_of[cell_id] = record["cell_hash"]
    rows = aggregate_by_hash(per_cell, hash_of, {}, {})
    return rows, hash_of


def diff_runs(
    dir_a: Path, dir_b: Path, metrics: list[str] | None = None
) -> list[dict[str, Any]]:
    """Join run A and run B on cell_hash; return per-cell delta rows."""
    metrics = metrics or DEFAULT_METRICS
    rows_a, _ = _collect(Path(dir_a))
    rows_b, _ = _collect(Path(dir_b))
    out: list[dict[str, Any]] = []
    for cell_hash in sorted(set(rows_a) & set(rows_b)):
        ra, rb = rows_a[cell_hash], rows_b[cell_hash]
        deltas: dict[str, float | None] = {}
        percents: dict[str, float | None] = {}
        for metric in metrics:
            va, vb = ra.mean.get(metric), rb.mean.get(metric)
            if va is None or vb is None:
                deltas[metric] = None
                percents[metric] = None
            else:
                deltas[metric] = round(vb - va, 2)
                percents[metric] = round((vb - va) / va * 100, 1) if va else None
        out.append(
            {
                "cell_hash": cell_hash,
                "cell_id_a": ra.cell_ids[0] if ra.cell_ids else None,
                "cell_id_b": rb.cell_ids[0] if rb.cell_ids else None,
                "a": {m: ra.mean.get(m) for m in metrics},
                "b": {m: rb.mean.get(m) for m in metrics},
                "delta": deltas,
                "delta_pct": percents,
            }
        )
    return out


def render_compare_md(diffs: list[dict[str, Any]], metrics: list[str]) -> str:
    lines = [
        "# Regression diff",
        "",
        "| cell_hash | cell_id_a | cell_id_b | "
        + " | ".join(f"{m} a→b (Δ%)" for m in metrics)
        + " |",
        "|" + "---|" * (3 + len(metrics)),
    ]
    for diff in diffs:
        cells = []
        for metric in metrics:
            a = diff["a"].get(metric)
            b = diff["b"].get(metric)
            pct = diff["delta_pct"].get(metric)
            a_txt = None if a is None else round(a, 2)
            b_txt = None if b is None else round(b, 2)
            cells.append(
                f"{a_txt}→{b_txt} ({pct:+.1f}%)"
                if pct is not None
                else f"{a_txt}→{b_txt} (-)"
            )
        lines.append(
            f"| `{diff['cell_hash']}` | `{diff['cell_id_a']}` "
            f"| `{diff['cell_id_b']}` | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def run(dir_a: Path, dir_b: Path, metrics: list[str] | None = None) -> Path:
    metrics = metrics or DEFAULT_METRICS
    diffs = diff_runs(dir_a, dir_b, metrics)
    out_path = Path(dir_b) / f"compare-{time.strftime('%Y%m%d-%H%M%S')}.md"
    out_path.write_text(render_compare_md(diffs, metrics), encoding="utf-8")
    payload = json.dumps(diffs, indent=2, ensure_ascii=False)
    out_path.with_suffix(".json").write_text(payload, encoding="utf-8")
    return out_path
