"""Merge stat-mode pipeline-stage exports for static DeepSeek EPLB.

Run on CPU with PyTorch installed, passing one file per stage from the same
recording session and dump request:

    python scripts/merge_eplb_records.py stage0.pt stage1.pt -o merged.pt

Load the result with --init-expert-location merged.pt and --eplb-algorithm
deepseek (or deepseek_hierarchical). The top-level logical_count contains
cumulative counts, not aligned forward records. It is not a replacement for
the per-record input used by deepseek_vec. Original stage payloads are retained
under pipeline_stages, including their separate utilization measurements.

Exports have no shared capture ID, so the caller must select files from the
same capture. Stage and tensor checks cannot detect every mixed-session input.
Input files are kept and an existing output file is never overwritten.
"""

import argparse
from pathlib import Path

import torch


def merge_stage_records(paths):
    stages = {}
    total = None
    occupied_layers = None

    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(record, dict):
            raise ValueError(f"{path}: expected a stat-mode export dictionary")
        stage = record.get("pipeline_parallel_rank")
        size = record.get("pipeline_parallel_size")
        if (
            type(size) is not int
            or size != len(paths)
            or type(stage) is not int
            or not 0 <= stage < size
        ):
            raise ValueError(
                f"{path}: expected one file for each of {size} pipeline stages; "
                f"got {len(paths)} files and stage {stage}"
            )
        if stage in stages:
            raise ValueError(f"{path}: duplicate pipeline stage {stage}")

        counts = record.get("logical_count")
        if (
            not isinstance(counts, torch.Tensor)
            or counts.ndim != 3
            or counts.shape[1] == 0
            or counts.shape[2] == 0
            or counts.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                f"{path}: logical_count must be an int32/int64 tensor "
                "of shape [records, global_layers, logical_experts]"
            )
        if (counts < 0).any():
            raise ValueError(f"{path}: negative expert counts")

        # Stage record indices need not align. Sum each stage independently;
        # int64 avoids overflowing int32 when combining a long recording.
        stage_total = counts.sum(dim=0, keepdim=True, dtype=torch.int64)
        active_layers = stage_total.ne(0).any(dim=2).squeeze(0)
        if total is None:
            total = torch.zeros_like(stage_total)
            occupied_layers = torch.zeros_like(active_layers)
        elif total.shape != stage_total.shape:
            raise ValueError(f"{path}: layer/expert dimensions differ across stages")
        if (occupied_layers & active_layers).any():
            raise ValueError(f"{path}: nonzero layer counts overlap another stage")
        occupied_layers |= active_layers
        total += stage_total
        stages[stage] = {**record, "source_file": path.name}

    if total is None:
        raise ValueError("Provide at least one pipeline-stage export")
    return {
        "logical_count": total,
        "count_aggregation": "sum_over_records",
        "pipeline_parallel_size": len(stages),
        "pipeline_stages": [stages[stage] for stage in range(len(stages))],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="One stat export per stage"
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        merged = merge_stage_records(args.inputs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    with args.output.open("xb") as output:
        torch.save(merged, output)
    print(
        f"Saved {args.output}: {len(merged['pipeline_stages'])} stages, "
        f"logical_count shape={tuple(merged['logical_count'].shape)}, "
        f"total={merged['logical_count'].sum().item()}"
    )


if __name__ == "__main__":
    main()
