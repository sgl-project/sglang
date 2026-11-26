"""Compare the three INT8 circular commit paths on SM120.

Run from the repo root with its CUDA environment:
    PYTHONPATH=python python test/manual/kernels/benchmark/attention/bench_gdn_replayssm_int8_commit.py
"""

import argparse
import json
import math
import statistics
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
)

LAYERS, BATCH, HEADS, VALUE_HEADS = 48, 12, 8, 24
HEAD_DIM, RING, HISTORY = 128, 32, 20
TRACK_SLOTS = 3
ARMS = ("ordinary_4dot", "ordinary_1dot", "persistent_1dot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("--samples must be positive and --warmup nonnegative")
    if torch.cuda.get_device_capability()[0] != 12:
        parser.error("this benchmark requires SM120-class CUDA hardware")

    torch.manual_seed(32)
    device = "cuda"

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device) * 0.1

    checkpoint = torch.randint(
        -127,
        128,
        (LAYERS, BATCH + TRACK_SLOTS, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
        device=device,
        dtype=torch.int8,
    )
    template = checkpoint.clone()
    scale = torch.full(
        (BATCH + TRACK_SLOTS, LAYERS, VALUE_HEADS, HEAD_DIM), 0.002, device=device
    )
    d = randn((LAYERS, BATCH, VALUE_HEADS, RING, HEAD_DIM))
    k = randn((LAYERS, BATCH, HEADS, RING, HEAD_DIM))
    g = -randn((LAYERS, BATCH, VALUE_HEADS, RING), torch.float32).abs()
    d_residual, k_residual = randn(d.shape) * 0.01, randn(k.shape) * 0.01
    rows = torch.arange(BATCH, device=device, dtype=torch.int32)
    write_pos = torch.empty_like(rows)
    cache_base = torch.empty_like(rows)
    flags = torch.empty_like(rows, dtype=torch.int8)
    accepted = torch.full_like(rows, 5)
    eviction = torch.empty(64 * 1024 * 1024, device=device)

    results = {}
    cases = [(n, 0) for n in (0, 1, 2, 3, 6, 12)] + [(0, TRACK_SLOTS)]
    for n_flush, n_track in cases:
        mask = torch.zeros_like(flags)
        positions = torch.linspace(0, BATCH - 1, n_flush, device=device).round().long()
        mask[positions] = 1
        track_args = {}
        if n_track:
            destinations = torch.full_like(rows, -1)
            steps = torch.full_like(rows, -1)
            tracked = rows[:: BATCH // n_track]
            destinations[tracked] = BATCH + rows[:n_track]
            steps[tracked] = 1
            track_args = dict(
                mamba_track_indices=destinations, mamba_steps_to_track=steps
            )

        def reset():
            checkpoint.copy_(template)
            scale.fill_(0.002)
            write_pos.fill_(HISTORY)
            cache_base.fill_(12)
            flags.copy_(mask)
            eviction.zero_()

        def commit(arm):
            residual = arm == "ordinary_4dot"
            capability = (
                patch("torch.cuda.get_device_capability", return_value=(10, 0))
                if arm == "ordinary_1dot"
                else nullcontext()
            )
            with capability:
                commit_gdn_replayssm_circular(
                    checkpoint_state=checkpoint,
                    checkpoint_scale=scale.transpose(0, 1),
                    d_cache=d,
                    k_cache=k,
                    g_cache=g,
                    d_residual_cache=d_residual if residual else None,
                    k_residual_cache=k_residual if residual else None,
                    state_batch_indices=rows,
                    replay_indices=rows,
                    write_pos=write_pos,
                    cache_base=cache_base,
                    is_flush=flags,
                    accept_lens=accepted,
                    **track_args,
                )

        graphs = {}
        for arm in ARMS:
            reset()
            commit(arm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                commit(arm)
            graphs[arm] = graph

        events = {arm: [] for arm in ARMS}
        for i in range(args.samples + args.warmup):
            # Rotate arm order to distribute clock and thermal drift.
            for arm in ARMS[i % 3 :] + ARMS[: i % 3]:
                reset()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[arm].replay()
                end.record()
                events[arm].append((start, end))

        torch.cuda.synchronize()
        case = f"{n_flush}/{BATCH} flush, {n_track} tracking"
        results[case] = {}
        for arm, pairs in events.items():
            us = sorted(
                start.elapsed_time(end) * 1_000 for start, end in pairs[args.warmup :]
            )
            results[case][arm] = {
                "p50_us": round(statistics.median(us), 2),
                "p90_us": round(us[math.ceil(0.9 * len(us)) - 1], 2),
            }

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "samples_per_arm": args.samples,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
