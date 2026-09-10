#!/usr/bin/env python
"""Run a configuration-driven benchmark sweep for SGLang (Ascend NPU aware).

Usage (from a sglang checkout, on the benchmark host):

    python benchmark/ascend_bench/run_sweep.py \
        --config benchmark/ascend_bench/configs/cookbook/qwen3-8b-bf16.yaml \
        --workdir /mnt/ascend_bench_runs
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asc_bench.config import ConfigError, load_config
from asc_bench.expand import expand_cells
from asc_bench.npu import hbm_used_mb
from asc_bench.provenance import capture
from asc_bench.report import load_provenance_header, render_report
from asc_bench.runner import Manifest, Runner
from asc_bench.sla import aggregate_by_hash, evaluate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML sweep config")
    parser.add_argument(
        "--workdir", default="./ascend_bench_runs", help="artifacts root"
    )
    parser.add_argument("--filter", help="glob pattern on cell_id to run a subset")
    parser.add_argument(
        "--run-id",
        default=None,
        help="resume an interrupted run: pass the run_id printed by the "
        "original invocation to reuse its directory and skip finished cells",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run cells whose last status was failed_*",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="expand and list cells, then exit"
    )
    parser.add_argument(
        "--skip-hbm-gate",
        action="store_true",
        help="disable the npu-smi HBM-free hard gate (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    cells = expand_cells(cfg)
    if args.filter:
        cells = [c for c in cells if fnmatch.fnmatch(c.cell_id, args.filter)]
    print(f"config {cfg.name}: {len(cells)} cell(s) after expansion/filter")

    if args.dry_run:
        for cell in cells:
            print(
                f"  {cell.cell_id}  hash={cell.cell_hash}  tp={cell.tp_size} "
                f"conc={cell.concurrency} num_prompts={cell.num_prompts}"
            )
        return 0

    run_id = args.run_id or f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.workdir) / run_id
    resumed = (run_dir / "manifest.jsonl").exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id: {run_id}  workdir: {run_dir}" + ("  (resume)" if resumed else ""))

    probe = None if args.skip_hbm_gate else hbm_used_mb
    runner = Runner(cfg, run_dir, hbm_probe=probe, log=print)
    runner.manifest.header(
        {
            "run_id": run_id,
            "config": cfg.name,
            "config_path": str(Path(args.config).resolve()),
            "provenance": capture(cfg.run.python),
        }
    )

    manifest_path = run_dir / "manifest.jsonl"
    done_before = Manifest.last_status_by_cell(manifest_path)
    exit_code = 0
    try:
        for cell in cells:
            previous = done_before.get(cell.cell_id)
            if previous == "done":
                print(f"[{cell.cell_id}] skip (already done)")
                continue
            if previous and previous.startswith("failed_") and not args.retry_failed:
                print(f"[{cell.cell_id}] skip ({previous}; use --retry-failed)")
                continue
            runner.run_cell(cell)
    except KeyboardInterrupt:
        print("interrupted; manifest allows resume with the same command", flush=True)
        exit_code = 130
    finally:
        records = Manifest.result_rows(manifest_path)
        per_cell = {
            r["cell_id"]: r.get("metrics") or {} for r in records if r.get("metrics")
        }
        hash_of = {c.cell_id: c.cell_hash for c in cells}
        accuracy_of = {
            r["cell_id"]: r.get("gsm8k_accuracy")
            for r in records
            if r.get("gsm8k_accuracy") is not None
        }
        cells_by_id = {c.cell_id: c for c in cells}
        rows = aggregate_by_hash(per_cell, hash_of, accuracy_of, cells_by_id)
        rows = evaluate(
            rows,
            cfg.sla,
            accuracy_floor=cfg.run.gsm8k.accuracy_floor if cfg.run.gsm8k else None,
        )
        provenance = load_provenance_header(manifest_path)
        render_report(
            run_dir, cfg.name, run_id, rows, records, provenance, cfg.sla.thresholds
        )
        statuses = Manifest.last_status_by_cell(manifest_path)
        terminal = [statuses.get(c.cell_id) for c in cells]
        if exit_code == 0 and any(s != "done" for s in terminal):
            exit_code = 1
        print(f"report: {run_dir / 'report.md'}")
    runner.manifest.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
