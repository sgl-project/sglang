"""SLA evaluation: parse bench output, aggregate repeats, filter, rank."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asc_bench.config import SLASpec

RANK_METRIC = "output_throughput"
TIEBREAK_METRIC = "p99_tpot_ms"
DEFAULT_CV_MAX = 0.15


@dataclass
class Row:
    """Aggregated result for one logical cell (same cell_hash)."""

    cell_hash: str
    cell_ids: list[str]
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    repeats: int = 0
    sla_pass: bool | None = None
    unrankable: bool = False
    unrankable_reason: str | None = None
    accuracy: float | None = None
    accuracy_ok: bool | None = None
    tp_size: int | None = None
    rank: int | None = None
    cell: Any = None  # representative Cell for rendering commands


def parse_last_metrics(path: str | Path) -> dict[str, Any] | None:
    """Return the last record with throughput metrics from a bench JSONL."""
    path = Path(path)
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and RANK_METRIC in record:
                last = record
    return last


def _cv(mean: float, std: float) -> float:
    return std / mean if mean else math.inf


def aggregate_by_hash(
    per_cell: dict[str, dict[str, Any]],
    hash_of_cell: dict[str, str],
    accuracy_of_cell: dict[str, float | None],
    cells_by_id: dict[str, Any],
) -> dict[str, Row]:
    """Group single-run metrics by cell_hash into aggregated rows."""
    groups: dict[str, list[str]] = {}
    for cell_id, cell_hash in hash_of_cell.items():
        groups.setdefault(cell_hash, []).append(cell_id)

    rows: dict[str, Row] = {}
    for cell_hash, cell_ids in groups.items():
        runs = [
            per_cell[cid]
            for cid in sorted(cell_ids)
            if cid in per_cell and per_cell[cid]
        ]
        representative = cells_by_id.get(sorted(cell_ids)[0])
        row = Row(
            cell_hash=cell_hash,
            cell_ids=sorted(cell_ids),
            repeats=len(runs),
            cell=representative,
        )
        row.tp_size = getattr(representative, "tp_size", None)
        accuracies = [
            accuracy_of_cell[cid]
            for cid in sorted(cell_ids)
            if cid in accuracy_of_cell and accuracy_of_cell[cid] is not None
        ]
        if accuracies:
            row.accuracy = statistics.mean(accuracies)
        if runs:
            keys = set().union(*(run.keys() for run in runs))
            for key in keys:
                # non-finite values (e.g. request_rate: Infinity in the
                # bench JSONL) crash statistics.stdev on py3.11 — skip them
                values = [
                    value
                    for run in runs
                    if isinstance((value := run.get(key)), (int, float))
                    and math.isfinite(value := float(value))
                ]
                if not values:
                    continue
                row.mean[key] = statistics.mean(values)
                row.std[key] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows[cell_hash] = row
    return rows


def evaluate(
    rows: dict[str, Row], sla: SLASpec, accuracy_floor: float | None = None
) -> list[Row]:
    """Fill sla_pass / unrankable / rank; return rows sorted for display.

    ``accuracy_floor`` comes from ``run.gsm8k.accuracy_floor``; rows with
    measured accuracy below the floor are excluded from ranking.
    """
    for row in rows.values():
        checks = []
        for key, threshold in sla.thresholds.items():
            if key not in row.mean:
                checks.append(False)
                continue
            checks.append(row.mean[key] <= threshold)
        # no thresholds configured -> the SLA gate passes vacuously
        row.sla_pass = all(checks) if checks else True

        if row.repeats > 1:
            for key in (RANK_METRIC, TIEBREAK_METRIC, *sla.thresholds):
                if key in row.mean and row.mean[key]:
                    cv = _cv(row.mean[key], row.std[key])
                    if cv > sla.cv_max:
                        row.unrankable = True
                        row.unrankable_reason = f"{key} cv={cv:.2f}>{sla.cv_max}"
                        break
        if row.accuracy is not None and accuracy_floor is not None:
            row.accuracy_ok = row.accuracy >= accuracy_floor

    eligible = [
        row
        for row in rows.values()
        if row.sla_pass and not row.unrankable and row.accuracy_ok is not False
    ]
    eligible.sort(
        key=lambda row: (
            -row.mean.get(RANK_METRIC, math.inf),
            row.mean.get(TIEBREAK_METRIC, math.inf),
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row.rank = rank
    return sorted(
        rows.values(),
        key=lambda row: (
            row.rank is None,
            row.rank if row.rank is not None else 0,
        ),
    )


def throughput_per_card(row: Row) -> float | None:
    if row.mean.get(RANK_METRIC) is None:
        return None
    tp = row.tp_size or 1
    return row.mean[RANK_METRIC] / tp
