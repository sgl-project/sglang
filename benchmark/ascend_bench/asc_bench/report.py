"""Render run artifacts: report.md (human) + report.json (machine)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from asc_bench.runner import stderr_tail
from asc_bench.sla import Row, throughput_per_card

DISPLAY_METRICS = [
    "output_throughput",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "mean_ttft_ms",
]


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value == int(value) and digits == 1:
        return str(int(value))
    return f"{value:.{digits}f}"


def _server_argv_text(row: Row) -> str:
    cell = row.cell
    if cell is None:
        return ""
    from asc_bench.config import argv_from_args

    return " ".join(argv_from_args(cell.server_args))


def _bench_argv_text(row: Row, cfg_name: str) -> str:
    cell = row.cell
    if cell is None:
        return ""
    parts = [
        "python -m sglang.bench_serving",
        "--backend sglang-oai",
        f"--dataset-name {cell.dataset_name}",
    ]
    from asc_bench.config import argv_from_args

    parts.append(" ".join(argv_from_args(cell.workload_args)))
    if cell.num_prompts:
        parts.append(f"--num-prompts {cell.num_prompts}")
    return " ".join(parts)


def render_report(
    run_dir: Path,
    config_name: str,
    run_id: str,
    rows: list[Row],
    cell_records: list[dict[str, Any]],
    provenance: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    """Write report.md and report.json into run_dir."""
    run_dir = Path(run_dir)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    done = sum(1 for r in cell_records if r.get("status") == "done")
    failed: dict[str, int] = {}
    for record in cell_records:
        status = record.get("status", "")
        if status.startswith("failed_"):
            failed[status] = failed.get(status, 0) + 1

    ranked = [row for row in rows if row.rank is not None]
    lines: list[str] = []
    lines.append(f"# SGLang Ascend benchmark report — {config_name}")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- generated_at: {generated_at}")
    summary = f"- cells: {len(cell_records)} total, {done} done"
    if failed:
        summary += ", " + ", ".join(f"{k}={v}" for k, v in sorted(failed.items()))
    lines.append(summary)
    lines.append("")
    lines.append("## Ranking (SLA-passing, variance-clean cells only)")
    lines.append("")
    header = (
        "| rank | cell_id | acc | tp | output_tok/s | tok/s/card | "
        + " | ".join(DISPLAY_METRICS[1:])
        + " |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(DISPLAY_METRICS)))
    for row in rows:
        rank = str(row.rank) if row.rank is not None else "-"
        acc = _fmt(row.accuracy, 3) if row.accuracy is not None else "-"
        per_card = throughput_per_card(row)
        cells_values = [_fmt(row.mean.get(metric)) for metric in DISPLAY_METRICS[1:]]
        lines.append(
            f"| {rank} | `{row.cell_ids[0] if row.cell_ids else '-'}` | {acc} "
            f"| {row.tp_size or '-'} "
            f"| {_fmt(row.mean.get('output_throughput'))} "
            f"| {_fmt(per_card)} | " + " | ".join(cells_values) + " |"
        )
    lines.append("")
    unrank = [row for row in rows if row.unrankable]
    if unrank:
        lines.append("## Unrankable (variance gate)")
        lines.append("")
        for row in unrank:
            lines.append(
                f"- `{row.cell_ids[0]}`: {row.unrankable_reason} "
                f"(repeats={row.repeats})"
            )
        lines.append("")

    lines.append("## SLA pass/fail matrix")
    lines.append("")
    lines.append("| cell_id | " + " | ".join(thresholds) + " | sla_pass |")
    lines.append("|" + "---|" * (len(thresholds) + 2))
    for row in rows:
        marks = []
        for key, threshold in thresholds.items():
            if key in row.mean:
                value = row.mean[key]
                mark = "ok" if value <= threshold else "FAIL"
                marks.append(f"{_fmt(value)} [{mark}]")
            else:
                marks.append("n/a")
        sla = "-" if row.sla_pass is None else ("pass" if row.sla_pass else "FAIL")
        lines.append(
            f"| `{row.cell_ids[0] if row.cell_ids else '-'}` | "
            + " | ".join(marks)
            + f" | {sla} |"
        )
    lines.append("")

    failed_records = [
        r for r in cell_records if r.get("status", "").startswith("failed_")
    ]
    if failed_records:
        lines.append("## Compatibility matrix (failed cells)")
        lines.append("")
        lines.append("| cell_id | status | detail |")
        lines.append("|---|---|---|")
        for record in failed_records:
            lines.append(
                f"| `{record['cell_id']}` | {record['status']} "
                f"| {record.get('detail') or '-'} |"
            )
        lines.append("")
        lines.append("<details><summary>server.log tails</summary>")
        lines.append("")
        for record in failed_records:
            tail = stderr_tail(run_dir / "cells" / record["cell_id"] / "server.log")
            if tail:
                lines.append(f"**{record['cell_id']}**")
                lines.append("```")
                lines.append(tail)
                lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    top = ranked[0] if ranked else None
    if top is not None:
        lines.append("## Recommended command (top-ranked cell)")
        lines.append("")
        lines.append("```bash")
        lines.append(f"# launch ({_server_argv_text(top)})")
        lines.append("python -m sglang.launch_server --model-path <model> \\")
        lines.append(f"  {_server_argv_text(top)}")
        lines.append("")
        lines.append(f"# benchmark ({_bench_argv_text(top, config_name)})")
        lines.append(_bench_argv_text(top, config_name))
        lines.append("```")
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    for key in (
        "sglang_version",
        "git_sha",
        "cann_version",
        "torch_npu_version",
        "npu_driver_version",
        "platform",
    ):
        lines.append(f"- {key}: `{provenance.get(key)}`")
    lines.append("")

    report_md = run_dir / "report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "schema": 1,
        "run_id": run_id,
        "config": config_name,
        "generated_at": generated_at,
        "provenance": provenance,
        "summary": {
            "total": len(cell_records),
            "done": done,
            "failed": failed,
        },
        "ranked": [
            {
                "rank": row.rank,
                "cell_id": row.cell_ids[0] if row.cell_ids else None,
                "cell_hash": row.cell_hash,
                "repeats": row.repeats,
                "sla_pass": row.sla_pass,
                "unrankable": row.unrankable,
                "accuracy": row.accuracy,
                "tp_size": row.tp_size,
                "mean": row.mean,
                "std": row.std,
                "output_throughput_per_card": throughput_per_card(row),
                "server_args": _server_argv_text(row),
            }
            for row in rows
        ],
        "cells": cell_records,
    }
    (run_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_provenance_header(manifest_path: Path) -> dict[str, Any]:
    if not Path(manifest_path).exists():
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "run_header":
                return record.get("provenance", {})
    return {}


def row_to_dict(row: Row) -> dict[str, Any]:
    return asdict(row)
