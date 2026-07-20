"""Micro-benchmark for the single-token (decode, M==1) fused MoE fast path.

Compares :func:`decode_single_moe` against the generic ``fused_experts`` path
for a few representative MoE shapes at ``num_tokens == 1``. Reports Triton-timed
median latency and the speedup.

Usage:
    python benchmark/kernels/fused_moe_triton/benchmark_moe_decode_single.py
"""

import torch
import triton

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_decode_single import (
    decode_single_moe,
)

DEVICE = "cuda"

# (name, E, top_k, hidden, intermediate)
SHAPES = [
    ("Qwen3-30B-A3B", 128, 8, 2048, 768),
    ("DeepSeek-V2-Lite", 64, 6, 2048, 1408),
]


def _make_inputs(E, topk, H, I):
    torch.manual_seed(0)
    hs = torch.randn(1, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(E, 2 * I, H, device=DEVICE, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, H, I, device=DEVICE, dtype=torch.bfloat16) * 0.05
    tid = torch.randperm(E, device=DEVICE)[:topk].unsqueeze(0).to(torch.int32)
    tw = torch.softmax(torch.randn(1, topk, device=DEVICE), dim=-1)
    return hs, w1, w2, tw, tid


def main():
    print(f"{'shape':20s} {'generic (us)':>14s} {'fast (us)':>12s} {'speedup':>9s}")
    for name, E, topk, H, I in SHAPES:
        hs, w1, w2, tw, tid = _make_inputs(E, topk, H, I)

        def run_generic():
            return fused_experts(
                hs, w1, w2, tw, tid, inplace=False, activation="silu"
            )

        def run_fast():
            return decode_single_moe(hs, w1, w2, tw, tid, 1.0)

        t_gen = triton.testing.do_bench(run_generic, warmup=50, rep=200)
        t_fast = triton.testing.do_bench(run_fast, warmup=50, rep=200)
        print(
            f"{name:20s} {t_gen*1e3:14.2f} {t_fast*1e3:12.2f} {t_gen/t_fast:8.3f}x"
        )


if __name__ == "__main__":
    main()
