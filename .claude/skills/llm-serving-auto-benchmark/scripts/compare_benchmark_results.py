#!/usr/bin/env python3
"""Summarize normalized cross-framework benchmark JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _float(row: dict[str, Any], path: str, default: float = 0.0) -> float:
    value = _get(row, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(row: dict[str, Any], path: str, default: bool = False) -> bool:
    value = _get(row, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _mean_ttft_ms(row: dict[str, Any]) -> float:
    return _float(row, "metrics.mean_ttft_ms", 1e30)


def _mean_tpot_ms(row: dict[str, Any]) -> float:
    return _float(row, "metrics.mean_tpot_ms", 1e30)


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _get(row, "status") == "ok",
        _bool(row, "sla.passed"),
        _float(row, "metrics.request_throughput"),
        _float(row, "metrics.output_token_throughput"),
        -_mean_ttft_ms(row),
        -_mean_tpot_ms(row),
        -_float(row, "hardware.gpu_count", 1e30),
    )


def _is_winner_candidate(row: dict[str, Any]) -> bool:
    return _get(row, "status") == "ok" and _bool(row, "sla.passed")


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value: Any, digits: int = 2) -> str:
    text = _fmt(value, digits)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _scenario(row: dict[str, Any]) -> str:
    for path in (
        "workload.scenario",
        "workload.scenario_name",
        "workload.dataset_scenario",
        "workload.dataset_name",
        "workload.kind",
        "scenario",
    ):
        value = _get(row, path)
        if value:
            return str(value)
    return "default"


def _server_command(row: dict[str, Any]) -> str:
    return str(_get(row, "server_command") or _get(row, "launch_command") or "")


def _artifact_summary(row: dict[str, Any]) -> str:
    artifacts = _get(row, "artifacts", {})
    if not isinstance(artifacts, dict):
        return ""
    parts = []
    for key in ("raw_result", "server_log", "benchmark_log", "summary"):
        value = artifacts.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "<br>".join(parts)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def best_by_framework_and_scenario(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not _is_winner_candidate(row):
            continue
        key = (str(_get(row, "framework", "unknown")), _scenario(row))
        if key not in best or _rank_key(row) > _rank_key(best[key]):
            best[key] = row
    return sorted(
        best.values(), key=lambda row: (_scenario(row), _rank_key(row)), reverse=True
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "framework",
        "scenario",
        "candidate_id",
        "status",
        "sla_passed",
        "request_throughput",
        "output_token_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "gpu_count",
        "server_command",
        "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": _get(row, "framework", ""),
                    "scenario": _scenario(row),
                    "candidate_id": _get(row, "candidate_id", ""),
                    "status": _get(row, "status", ""),
                    "sla_passed": _bool(row, "sla.passed"),
                    "request_throughput": _get(row, "metrics.request_throughput", ""),
                    "output_token_throughput": _get(
                        row, "metrics.output_token_throughput", ""
                    ),
                    "mean_ttft_ms": _get(row, "metrics.mean_ttft_ms", ""),
                    "mean_tpot_ms": _get(row, "metrics.mean_tpot_ms", ""),
                    "p99_ttft_ms": _get(row, "metrics.p99_ttft_ms", ""),
                    "p99_tpot_ms": _get(row, "metrics.p99_tpot_ms", ""),
                    "gpu_count": _get(row, "hardware.gpu_count", ""),
                    "server_command": _server_command(row),
                    "failure_reason": _get(row, "failure_reason", ""),
                }
            )


def _append_best_commands_by_framework(
    lines: list[str], scenario_winners: list[dict[str, Any]]
) -> None:
    frameworks = sorted(
        {str(_get(row, "framework", "unknown")) for row in scenario_winners}
    )
    lines.extend(["## Best Commands By Framework", ""])
    for framework in frameworks:
        lines.extend(
            [
                f"### `{framework}`",
                "",
                "| Scenario | Candidate | Status | SLA | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Success rate | GPUs | Server command | Artifacts |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        rows = [row for row in scenario_winners if _get(row, "framework") == framework]
        for row in sorted(rows, key=_scenario):
            lines.append(
                "| {scenario} | {candidate} | {status} | {sla} | {rps} | {otps} | {ttps} | {ttft} | {tpot} | {success} | {gpus} | {command} | {artifacts} |".format(
                    scenario=_cell(_scenario(row)),
                    candidate=_cell(_get(row, "candidate_id", "")),
                    status=_cell(_get(row, "status", "")),
                    sla=_cell(_bool(row, "sla.passed")),
                    rps=_cell(_get(row, "metrics.request_throughput")),
                    otps=_cell(_get(row, "metrics.output_token_throughput")),
                    ttps=_cell(_get(row, "metrics.total_token_throughput")),
                    ttft=_cell(_get(row, "metrics.mean_ttft_ms")),
                    tpot=_cell(_get(row, "metrics.mean_tpot_ms")),
                    success=_cell(_get(row, "metrics.success_rate")),
                    gpus=_cell(_get(row, "hardware.gpu_count")),
                    command=_cell(_server_command(row)),
                    artifacts=_cell(_artifact_summary(row)),
                )
            )
        lines.append("")


def _append_cross_framework_table(
    lines: list[str], scenario_winners: list[dict[str, Any]]
) -> None:
    lines.extend(
        [
            "## Cross-Framework Best Comparison",
            "",
            "| Scenario | Rank | Framework | Candidate | SLA | Req/s | Output tok/s | Mean TTFT ms | Mean TPOT ms | GPUs | Server command |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    scenario_names = sorted({_scenario(row) for row in scenario_winners})
    for scenario_name in scenario_names:
        rows = [row for row in scenario_winners if _scenario(row) == scenario_name]
        for rank, row in enumerate(sorted(rows, key=_rank_key, reverse=True), 1):
            lines.append(
                "| {scenario} | {rank} | {framework} | {candidate} | {sla} | {rps} | {otps} | {ttft} | {tpot} | {gpus} | {command} |".format(
                    scenario=_cell(scenario_name),
                    rank=rank,
                    framework=_cell(_get(row, "framework", "")),
                    candidate=_cell(_get(row, "candidate_id", "")),
                    sla=_cell(_bool(row, "sla.passed")),
                    rps=_cell(_get(row, "metrics.request_throughput")),
                    otps=_cell(_get(row, "metrics.output_token_throughput")),
                    ttft=_cell(_get(row, "metrics.mean_ttft_ms")),
                    tpot=_cell(_get(row, "metrics.mean_tpot_ms")),
                    gpus=_cell(_get(row, "hardware.gpu_count")),
                    command=_cell(_server_command(row)),
                )
            )
    lines.append("")


def render_markdown(rows: list[dict[str, Any]]) -> str:
    scenario_winners = best_by_framework_and_scenario(rows)

    lines = ["# Benchmark Summary", ""]
    if not rows:
        lines.append("No rows found.")
        return "\n".join(lines) + "\n"

    _append_best_commands_by_framework(lines, scenario_winners)
    _append_cross_framework_table(lines, scenario_winners)

    failed = [
        row
        for row in rows
        if _get(row, "status") != "ok" or not _bool(row, "sla.passed")
    ]
    if failed:
        lines.extend(
            [
                "",
                "## Failed Or SLA-Failing Candidates",
                "",
                "This table records tried configs that were not selected. They either failed, were skipped by policy, or completed without passing the SLA.",
                "",
                "| Framework | Candidate | Status | SLA | Reason |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in failed:
            lines.append(
                "| {framework} | {candidate} | {status} | {sla} | {reason} |".format(
                    framework=_cell(_get(row, "framework", "")),
                    candidate=_cell(_get(row, "candidate_id", "")),
                    status=_cell(_get(row, "status", "")),
                    sla=_cell(_bool(row, "sla.passed")),
                    reason=_cell(_get(row, "failure_reason", "")),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Normalized JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Markdown summary")
    parser.add_argument("--csv", type=Path, help="Optional CSV table")
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, sorted(rows, key=_rank_key, reverse=True))


if __name__ == "__main__":
    main()
