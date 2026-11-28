"""Compare BF16 and INT8 GDN ReplaySSM verify across the SM120 tile boundary."""

import argparse
import json
import math
import statistics
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)

LAYERS, HEADS, VALUE_HEADS, DIM = 48, 8, 24, 128
RING, WINDOW = 32, 8
ARMS = ("bf16_fixed", "int8_fixed", "int8_sm120")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches", nargs="+", type=int, default=(1, 8, 16, 23, 24, 25, 32)
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or any(b < 1 for b in args.batches):
        parser.error("batch sizes and samples must be positive, and warmup nonnegative")
    if torch.cuda.get_device_capability()[0] != 12:
        parser.error("this benchmark requires SM120-class CUDA hardware")

    torch.manual_seed(53223)
    device = "cuda"

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device) * 0.1

    slots = max(args.batches)
    shape = (LAYERS, slots, VALUE_HEADS, DIM, DIM)
    state_i8 = torch.randint(-127, 128, shape, device=device, dtype=torch.int8)
    state_scale = (
        torch.rand((slots, LAYERS, VALUE_HEADS, DIM), device=device) * 0.005 + 0.001
    ).transpose(0, 1)
    state_bf16 = torch.empty_like(state_i8, dtype=torch.bfloat16)
    for layer in range(LAYERS):
        state_bf16[layer] = state_i8[layer].float() * state_scale[layer].unsqueeze(-1)
    d = randn((LAYERS, slots, VALUE_HEADS, RING, DIM))
    k = randn((LAYERS, slots, HEADS, RING, DIM))
    g = -randn((LAYERS, slots, VALUE_HEADS, RING), torch.float32).abs()
    d_low, k_low = randn(d.shape) * 0.01, randn(k.shape) * 0.01
    eviction = torch.empty(64 * 1024 * 1024, device=device)
    results = {}

    for batch in args.batches:
        tokens = batch * WINDOW
        query, key = (randn((LAYERS, tokens, HEADS, DIM)) for _ in range(2))
        value = randn((LAYERS, tokens, VALUE_HEADS, DIM))
        a, b = (randn((LAYERS, tokens, VALUE_HEADS)) for _ in range(2))
        A_log = randn((LAYERS, VALUE_HEADS), torch.float32) - 1
        dt_bias = randn((LAYERS, VALUE_HEADS), torch.float32)
        out = torch.empty_like(value)
        rows = torch.arange(batch, device=device, dtype=torch.int32)
        starts = torch.arange(0, tokens + 1, WINDOW, device=device, dtype=torch.int32)
        flush = torch.zeros(batch, device=device, dtype=torch.int8)

        for history in (0, 20):
            positions = torch.full_like(rows, history)
            bases = torch.full_like(rows, 24 if history else 0)

            def run(arm):
                bf16 = arm == "bf16_fixed"
                state = state_bf16 if bf16 else state_i8
                capability = (
                    patch("torch.cuda.get_device_capability", return_value=(10, 0))
                    if arm == "int8_fixed"
                    else nullcontext()
                )
                with capability:
                    for layer in range(LAYERS):
                        gdn_replayssm_spec_decode(
                            q=query[layer],
                            k=key[layer],
                            v=value[layer],
                            a=a[layer],
                            b=b[layer],
                            A_log=A_log[layer],
                            dt_bias=dt_bias[layer],
                            checkpoint_state=state[layer],
                            checkpoint_scale=None if bf16 else state_scale[layer],
                            d_cache=d[layer],
                            k_cache=k[layer],
                            g_cache=g[layer],
                            rawv_cache=d_low[layer],
                            rawk_cache=k_low[layer],
                            beta_cache=None,
                            out=out[layer],
                            query_start_loc=starts,
                            ssm_state_indices=rows,
                            replay_indices=rows,
                            write_pos=positions,
                            cache_base=bases,
                            is_flush=flush,
                            max_cache_len=RING,
                            max_spec_len=WINDOW,
                            block_v=64,
                            num_warps=2,
                            num_stages=1,
                            nk=2,
                            launch_mode="verify",
                            null_block_id=-1,
                        )

            graphs = {}
            for arm in ARMS:
                run(arm)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(arm)
                graphs[arm] = graph

            events = {arm: [] for arm in ARMS}
            for i in range(args.samples + args.warmup):
                for arm in ARMS[i % len(ARMS) :] + ARMS[: i % len(ARMS)]:
                    eviction.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[arm].replay()
                    end.record()
                    events[arm].append((start, end))
            torch.cuda.synchronize()
            results[f"B{batch}_H{history}"] = {}
            for arm, pairs in events.items():
                us = sorted(
                    start.elapsed_time(end) * 1000
                    for start, end in pairs[args.warmup :]
                )
                results[f"B{batch}_H{history}"][arm] = {
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
