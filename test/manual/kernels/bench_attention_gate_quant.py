"""Check and time static FP8 attention output gating."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8


def timing(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(9):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        for _ in range(10):
            graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--quant-only", action="store_true")
    args = parser.parse_args()
    records = []
    torch.manual_seed(42)
    for m in (1, 2, 7, 14):
        for heads in (4, 8, 16):
            hidden = heads * 256
            x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
            gate = torch.randn(m, heads, 512, dtype=torch.bfloat16, device="cuda")[
                :, :, :256
            ]
            for scale_value in (0.001, 0.01, 0.1, 1.0):
                scale = torch.tensor([scale_value], device="cuda", dtype=torch.float32)

                def stock():
                    y = fused_sigmoid_mul(x, gate)
                    return static_quant_fp8(y, scale)[0]

                def fused():
                    return fused_sigmoid_mul(x, gate, static_scale=scale)

                expected, actual = stock(), fused()
                mismatches = (
                    (expected.view(torch.uint8) != actual.view(torch.uint8))
                    .sum()
                    .item()
                )
                assert mismatches == 0, (m, heads, scale_value, mismatches)
            stock_us, fused_us = (
                ([], []) if args.check_only else (timing(stock), timing(fused))
            )
            gemm_bytes_equal = None
            if not args.quant_only:
                from sglang.srt.layers.quantization.fp8_utils import (
                    apply_fp8_linear_bmm_flashinfer,
                )

                gated = fused_sigmoid_mul(x, gate)
                quantized = fused_sigmoid_mul(x, gate, static_scale=scale)
                weight = torch.randn(hidden, 4096, device="cuda").to(
                    torch.float8_e4m3fn
                )
                weight_scale = torch.tensor([0.02], device="cuda")
                old = apply_fp8_linear_bmm_flashinfer(
                    gated, weight, weight_scale, scale
                )
                new = apply_fp8_linear_bmm_flashinfer(
                    quantized, weight, weight_scale, scale
                )
                torch.testing.assert_close(old, new, rtol=0, atol=0)
                gemm_bytes_equal = torch.equal(
                    old.view(torch.uint8), new.view(torch.uint8)
                )
            row = dict(
                m=m,
                heads=heads,
                hidden=hidden,
                stock_us=stock_us,
                fused_us=fused_us,
                stock_median_us=statistics.median(stock_us) if stock_us else None,
                fused_median_us=statistics.median(fused_us) if fused_us else None,
                bytes_equal=True,
                gemm_bytes_equal=gemm_bytes_equal,
                numerical_verdict="PASS",
                end_to_end_verdict="PENDING",
            )
            records.append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    dict(
                        records=records,
                        script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
