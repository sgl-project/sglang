"""Benchmark fused Kimi MLA value projection and MXFP4 output quantization."""

import argparse

import torch
import triton
from aiter.ops.triton.quant import dynamic_mxfp4_quant

from sglang.srt.layers.quantization.rocm_mla_value_bmm_mxfp4 import (
    batched_gemm_a16wfp4_flatten_mxfp4_quant,
)
from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
    batched_gemm_afp4wfp4_pre_quant,
    fused_flatten_mxfp4_quant,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    args = parser.parse_args()

    n, k = 128, 512
    weight = torch.randn((args.heads * n, k), dtype=torch.bfloat16, device="cuda")
    weight_fp4, weight_scales = dynamic_mxfp4_quant(weight)
    weight_fp4 = weight_fp4.view(args.heads, n, k // 2)
    weight_scales = weight_scales.view(args.heads, n, k // 32)

    def split(x):
        bf16_output = torch.empty(
            (x.shape[1], args.heads, n), dtype=torch.bfloat16, device=x.device
        )
        batched_gemm_afp4wfp4_pre_quant(
            x,
            weight_fp4,
            weight_scales,
            torch.bfloat16,
            bf16_output.transpose(0, 1),
        )
        return fused_flatten_mxfp4_quant(bf16_output)

    print(f"{'tokens':>8} {'split (us)':>12} {'fused (us)':>12} {'speedup':>10}")
    for tokens in args.tokens:
        x = torch.randn((args.heads, tokens, k), dtype=torch.bfloat16, device="cuda")
        split_ms = triton.testing.do_bench(lambda: split(x))
        fused_ms = triton.testing.do_bench(
            lambda: batched_gemm_a16wfp4_flatten_mxfp4_quant(
                x, weight_fp4, weight_scales
            )
        )
        print(
            f"{tokens:>8} {split_ms * 1e3:>12.3f} "
            f"{fused_ms * 1e3:>12.3f} {split_ms / fused_ms:>10.3f}"
        )


if __name__ == "__main__":
    main()
