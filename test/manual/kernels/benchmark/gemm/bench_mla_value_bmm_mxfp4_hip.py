"""Benchmark fused Kimi MLA value projection and MXFP4 output quantization.

Each path replays one CUDA graph over rotated copies of x, W and W scales,
totaling ROTATE_BYTES so every call reads its inputs cold from HBM;
on MI355X that must exceed the 256 MB Infinity Cache, not only the 4 MB per-XCD L2.
Reports the median replay.

Run: ``python test/manual/kernels/benchmark/gemm/bench_mla_value_bmm_mxfp4_hip.py``
"""

import argparse
import math
import statistics

import torch
from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import (
    batched_gemm_afp4wfp4_pre_quant,
)
from aiter.ops.triton.fused_mxfp4_quant import fused_flatten_mxfp4_quant
from aiter.ops.triton.quant import dynamic_mxfp4_quant

from sglang.kernels.ops.gemm.mla_value_bmm_mxfp4_hip import (
    batched_gemm_a16wfp4_flatten_mxfp4_quant,
)

N, K = 128, 512
# 4x the MI355X 256 MB Infinity Cache.
ROTATE_BYTES = 1 << 30


def split(x, w, w_scales):
    heads, tokens, _ = x.shape
    bf16_output = torch.empty((tokens, heads, N), dtype=torch.bfloat16, device="cuda")
    batched_gemm_afp4wfp4_pre_quant(
        x, w, w_scales, torch.bfloat16, bf16_output.transpose(0, 1)
    )
    return fused_flatten_mxfp4_quant(bf16_output)


def make_inputs(heads, tokens):
    weight = torch.randn((heads * N, K), dtype=torch.bfloat16, device="cuda")
    w, w_scales = dynamic_mxfp4_quant(weight)
    w = w.view(heads, N, K // 2)
    w_scales = w_scales.view(heads, N, K // 32)
    x = torch.randn((heads, tokens, K), dtype=torch.bfloat16, device="cuda")
    nbytes = sum(t.numel() * t.element_size() for t in (x, w, w_scales))
    copies = math.ceil(ROTATE_BYTES / nbytes)
    return [(x.clone(), w.clone(), w_scales.clone()) for _ in range(copies)]


def capture(fn, inputs):
    for args in inputs[:3]:
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in inputs:
            fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_ms(graph):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def bench_us(fns, inputs, replays):
    graphs = [capture(fn, inputs) for fn in fns]
    # Interleave replays so every path sees the same background load.
    times_ms = [[] for _ in graphs]
    for _ in range(replays):
        for graph, times in zip(graphs, times_ms):
            times.append(time_ms(graph))
    del graphs
    torch.cuda.empty_cache()
    return [statistics.median(t) * 1e3 / len(inputs) for t in times_ms]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()

    print(f"{'tokens':>8} {'split (us)':>12} {'fused (us)':>12} {'speedup':>10}")
    for tokens in args.tokens:
        inputs = make_inputs(args.heads, tokens)
        split_us, fused_us = bench_us(
            (split, batched_gemm_a16wfp4_flatten_mxfp4_quant), inputs, args.replays
        )
        print(
            f"{tokens:>8} {split_us:>12.3f} {fused_us:>12.3f} "
            f"{split_us / fused_us:>10.3f}"
        )
        del inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
