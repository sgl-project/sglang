#!/usr/bin/env python3
"""Sweep SM100 Triton Q8KV8 tiles for the DeepSeek-V4-Flash TP8 shape."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm90 import (
    _sparse_mla_q8kv8_prefill_fwd_sm100_trusted,
)
from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm100 import (
    sparse_mla_q8kv8_prefill_fwd_sm100,
)


def _measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    )
    return values[len(values) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-q", type=int, default=512)
    parser.add_argument("--s-kv", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7" or torch.cuda.device_count() != 1:
        raise SystemExit("Safety check failed: expose physical GPU7 only")
    if torch.cuda.get_device_capability(0)[0] != 10:
        raise SystemExit("SM100 is required")

    generator = torch.Generator(device="cuda").manual_seed(42)
    q = (torch.randn(args.s_q, 8, 512, device="cuda", generator=generator) * 0.05).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(args.s_kv, 1, 512, device="cuda", generator=generator) * 0.05).to(
        torch.float8_e4m3fn
    )
    indices = torch.randint(
        0,
        args.s_kv,
        (args.s_q, 1, args.topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    sink = torch.zeros(8, dtype=torch.float32, device="cuda")
    lengths = torch.full((args.s_q,), args.topk, dtype=torch.int32, device="cuda")
    out = torch.empty(args.s_q, 8, 512, dtype=torch.bfloat16, device="cuda")
    max_logits = torch.empty(args.s_q, 8, dtype=torch.float32, device="cuda")
    lse = torch.empty_like(max_logits)

    max_logits_cuda = torch.empty(args.s_q, 32, dtype=torch.float32, device="cuda")
    lse_cuda = torch.empty_like(max_logits_cuda)

    def launch_cuda() -> None:
        _sparse_mla_q8kv8_prefill_fwd_sm100_trusted(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=1.0 / (512**0.5),
            q_scale=scale,
            kv_scale=scale,
            attn_sink=sink,
            topk_length=lengths,
            out=out,
            max_logits=max_logits_cuda,
            lse=lse_cuda,
            active_heads=8,
        )

    rows = [
        {
            "implementation": "cuda_tcgen05",
            "median_ms": _measure(launch_cuda, args.warmup, args.repeats),
        }
    ]
    print(rows[0], flush=True)

    os.environ["SGLANG_SM100_Q8KV8_TRITON"] = "1"
    for block_h in (8, 16, 32):
        for block_n in (64, 128, 256):
            for num_warps in (4, 8):
                os.environ["SGLANG_SM100_Q8KV8_BLOCK_H"] = str(block_h)
                os.environ["SGLANG_SM100_Q8KV8_BLOCK_N"] = str(block_n)
                os.environ["SGLANG_SM100_Q8KV8_NUM_WARPS"] = str(num_warps)
                os.environ["SGLANG_SM100_Q8KV8_NUM_STAGES"] = "1"

                def launch() -> None:
                    sparse_mla_q8kv8_prefill_fwd_sm100(
                        q=q,
                        kv=kv,
                        indices=indices,
                        sm_scale=1.0 / (512**0.5),
                        q_scale=scale,
                        kv_scale=scale,
                        attn_sink=sink,
                        topk_length=lengths,
                        out=out,
                        max_logits=max_logits,
                        lse=lse,
                    )

                try:
                    median_ms = _measure(launch, args.warmup, args.repeats)
                    row = {
                        "implementation": "triton",
                        "block_h": block_h,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "median_ms": median_ms,
                    }
                except Exception as error:
                    row = {
                        "implementation": "triton",
                        "block_h": block_h,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "error": str(error),
                    }
                rows.append(row)
                print(row, flush=True)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-triton-tile-sweep.json"
    path.write_text(
        json.dumps(
            {
                "arguments": vars(args) | {"results_dir": str(args.results_dir)},
                "rows": rows,
            },
            indent=2,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
