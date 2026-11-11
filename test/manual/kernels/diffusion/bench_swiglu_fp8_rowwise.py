# SPDX-License-Identifier: Apache-2.0
"""Compare packed SwiGLU + FP8 quantization with its standalone fusion.

Run on an idle GPU, e.g. CUDA_VISIBLE_DEVICES=2 python this_file.py
--output-json logs/swiglu-fp8.json. CUDA graph timing excludes Python overhead.
Each case measures both the producer alone and producer + unchanged FP8 GEMM.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.diffusion import (
    fp8_rowwise,
    fused_packed_silu_mul_bitexact,
    fused_packed_swiglu_fp8_rowwise,
)


def benchmark(rows, hidden, rounds, rep_ms):
    torch.manual_seed(42)
    storage = torch.randn((1, rows, 3 * hidden), device="cuda", dtype=torch.bfloat16)
    packed = storage[..., hidden:]
    weights, weight_scales = fp8_rowwise(
        torch.randn((3072, hidden), device="cuda", dtype=torch.bfloat16), 16
    )

    def separate():
        values = fused_packed_silu_mul_bitexact(packed)
        return fp8_rowwise(values.reshape(rows, hidden), 16)

    def fused():
        return fused_packed_swiglu_fp8_rowwise(packed)

    def gemm(producer):
        q, scales = producer()
        return torch._scaled_mm(
            q,
            weights.T,
            scales[:, None],
            weight_scales[None, :],
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    q, scales = separate()
    fq, fs = fused()
    assert torch.equal(q.view(torch.uint8), fq.view(torch.uint8))
    assert torch.equal(scales, fs)
    assert torch.equal(gemm(separate), gemm(fused))
    result = {
        "rows": rows,
        "hidden": hidden,
        "input_row_stride": packed.stride(1),
        "padded_rows": q.shape[0],
        "bit_exact": True,
    }
    for name, a, b in [
        ("producer", separate, fused),
        ("with_gemm", lambda: gemm(separate), lambda: gemm(fused)),
    ]:
        for _ in range(10):
            a()
            b()
        torch.cuda.synchronize()
        samples = {"separate": [], "fused": []}
        # Alternate order to reduce thermal/clock bias.
        for i in range(rounds):
            order = [("separate", a), ("fused", b)]
            if i % 2:
                order.reverse()
            for key, fn in order:
                samples[key].append(
                    triton.testing.do_bench_cudagraph(fn, rep=rep_ms) * 1000
                )
        medians = {k: statistics.median(v) for k, v in samples.items()}
        result[name] = {
            "samples_us": samples,
            "median_us": medians,
            "speedup": medians["separate"] / medians["fused"],
            "reduction_pct": 100 * (1 - medians["fused"] / medians["separate"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    if args.rounds <= 0 or args.rep_ms <= 0:
        parser.error("rounds and rep-ms must be positive")
    results = []
    for rows, hidden in [
        (1, 9216),
        (32, 9216),
        (340, 9216),
        (2720, 9216),
        (3173, 9216),
        (3173, 3072),
    ]:
        r = benchmark(rows, hidden, args.rounds, args.rep_ms)
        results.append(r)
        print(json.dumps(r), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "triton": triton.__version__,
                    "mode": "CUDA graph, no profiler",
                    "rounds": args.rounds,
                    "rep_ms": args.rep_ms,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
