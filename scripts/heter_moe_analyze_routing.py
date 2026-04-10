"""Analyze EPLB routing dumps and output per-layer per-expert counts.

Reads .pt files from ExpertDistributionRecorder, splits by forward_mode
(prefill vs decode), aggregates across TP ranks, and outputs JSON files
in the same format as the original collector:
  {"transformer_block_0": [128 ints], ..., "transformer_block_47": [128 ints]}

Usage:
  python3 scripts/heter_moe_analyze_routing.py \
      --input-dir /data/heter-moe/routing_stats/eplb \
      --output-dir /data/heter-moe/routing_stats/eplb_analyzed
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

NUM_EXPERTS = 128
NUM_LAYERS = 48


def load_and_group_dumps(input_dir):
    """Load rank-0 .pt files grouped by collection round.

    TP ranks have slightly different timestamps, so we filter rank=0 only.
    Each rank-0 file = one collection round (one batch size).
    """
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pt"))
    rank0_files = []
    for fname in files:
        parts = fname.replace("expert_distribution_recorder_", "").replace(".pt", "")
        ts, rank = parts.rsplit("_", 1)
        if rank == "0":
            rank0_files.append((ts, os.path.join(input_dir, fname)))
    return rank0_files


def aggregate_records(records, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS):
    """Sum global_physical_count across records → [num_layers, num_experts]."""
    total = np.zeros((num_layers, num_experts), dtype=np.int64)
    for r in records:
        counts = r["global_physical_count"].numpy()
        total[: counts.shape[0], : counts.shape[1]] += counts
    return total


def counts_to_json(counts):
    """Convert [num_layers, num_experts] array to the standard JSON format."""
    result = {}
    for layer_idx in range(counts.shape[0]):
        result[f"transformer_block_{layer_idx}"] = counts[layer_idx].tolist()
    return result


def print_summary(label, counts):
    avg = counts.mean(axis=0)
    print(
        f"  {label}: total_tokens={int(counts.sum())}, "
        f"max/expert={int(avg.max()):.0f}, min={int(avg.min()):.0f}, "
        f"max/mean={avg.max() / max(avg.mean(), 1):.1f}x, "
        f"zeros={int((avg == 0).sum())}/128"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=str, default="/data/heter-moe/routing_stats/eplb"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/data/heter-moe/routing_stats/eplb_analyzed"
    )
    args = parser.parse_args()

    rank0_files = load_and_group_dumps(args.input_dir)
    print(
        f"Found {len(rank0_files)} collection rounds (rank-0 only) in {args.input_dir}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for round_idx, (ts, filepath) in enumerate(rank0_files):
        d = torch.load(filepath, weights_only=False)
        records = d["records"]

        prefill_records = [r for r in records if r["forward_mode"] == 1]
        decode_records = [r for r in records if r["forward_mode"] == 2]

        print(
            f"\nRound {round_idx} (ts={ts}): "
            f"{len(records)} total, {len(prefill_records)} prefill, {len(decode_records)} decode"
        )

        if prefill_records:
            pf_counts = aggregate_records(prefill_records)
            pf_json = counts_to_json(pf_counts)
            pf_path = os.path.join(args.output_dir, f"round{round_idx}_prefill.json")
            with open(pf_path, "w") as f:
                json.dump(pf_json, f, indent=2)
            print_summary("prefill", pf_counts)
            print(f"    → {pf_path}")

        if decode_records:
            dc_counts = aggregate_records(decode_records)
            dc_json = counts_to_json(dc_counts)
            dc_path = os.path.join(args.output_dir, f"round{round_idx}_decode.json")
            with open(dc_path, "w") as f:
                json.dump(dc_json, f, indent=2)
            print_summary("decode", dc_counts)
            print(f"    → {dc_path}")

    print(f"\nAll analyzed stats saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
