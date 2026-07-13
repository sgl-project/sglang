#!/usr/bin/env python3
"""Build and validate the scheduled 40-request PD-flip experiment trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_trace(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 40:
        raise ValueError(f"expected 40 requests, got {len(rows)}")
    expected_kinds = ["long", "short"] * 20
    if [row.get("prompt_kind") for row in rows] != expected_kinds:
        raise ValueError("requests must alternate long, short")
    if sum(row.get("prompt_chars") == 10000 for row in rows) != 20:
        raise ValueError("expected 20 requests with prompt_chars=10000")
    if sum(row.get("prompt_chars") == 1000 for row in rows) != 20:
        raise ValueError("expected 20 requests with prompt_chars=1000")
    if not all(
        float(row.get("ttft_slo_s", 0)) > 0
        and float(row.get("tpot_slo_s", 0)) > 0
        for row in rows
    ):
        raise ValueError("every request needs positive TTFT and TPOT SLO values")


def prepare_trace(
    source: Path,
    output: Path,
    manifest: Path,
    wave_size: int,
    wave_gap_seconds: float,
    intra_wave_interval_seconds: float,
    ttft_slo_override_seconds: float = 0.0,
) -> None:
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    if wave_gap_seconds < 0 or intra_wave_interval_seconds < 0:
        raise ValueError("trace timing values must be non-negative")

    source_rows = _load_jsonl(source)
    _validate_trace(source_rows)
    scheduled_rows = []
    for index, row in enumerate(source_rows):
        scheduled = dict(row)
        scheduled["arrival_offset_s"] = (
            (index // wave_size) * wave_gap_seconds
            + (index % wave_size) * intra_wave_interval_seconds
        )
        if ttft_slo_override_seconds > 0:
            scheduled["ttft_slo_s"] = ttft_slo_override_seconds
        scheduled_rows.append(scheduled)

    if any(
        left["arrival_offset_s"] > right["arrival_offset_s"]
        for left, right in zip(scheduled_rows, scheduled_rows[1:])
    ):
        raise ValueError(
            "wave_gap_seconds is too small to keep arrival offsets monotonic"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in scheduled_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    reloaded = _load_jsonl(output)
    _validate_trace(reloaded)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "source_trace": str(source),
                "effective_trace": str(output),
                "source_sha256": _sha256(source),
                "effective_sha256": _sha256(output),
                "request_count": len(reloaded),
                "wave_size": wave_size,
                "wave_gap_seconds": wave_gap_seconds,
                "intra_wave_interval_seconds": intra_wave_interval_seconds,
                "ttft_slo_override_seconds": ttft_slo_override_seconds,
                "last_arrival_offset_s": reloaded[-1]["arrival_offset_s"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave-size", type=int, required=True)
    parser.add_argument("--wave-gap-seconds", type=float, required=True)
    parser.add_argument(
        "--intra-wave-interval-seconds", type=float, required=True
    )
    parser.add_argument("--ttft-slo-override-seconds", type=float, default=0.0)
    args = parser.parse_args()
    prepare_trace(
        source=args.source,
        output=args.output,
        manifest=args.manifest,
        wave_size=args.wave_size,
        wave_gap_seconds=args.wave_gap_seconds,
        intra_wave_interval_seconds=args.intra_wave_interval_seconds,
        ttft_slo_override_seconds=args.ttft_slo_override_seconds,
    )


if __name__ == "__main__":
    main()
