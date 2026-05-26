"""Targeted benchmark for the SM90 FP8 swap-AB dispatch path.

Sweeps small batch sizes (M = 1..128) across N/K shapes that exercise each
dispatch bucket in `fp8_gemm_sm90_dispatch.cuh`. Output style matches
`bench_fp8_gemm.py`: `triton.testing.perf_report` + GB/s table per (N, K).

Compare against `main` by:
  1. Run on `main`:        `python bench_fp8_gemm_swap_ab.py > main.txt`
  2. Run on feature branch:`python bench_fp8_gemm_swap_ab.py > swap_ab.txt`
  3. Diff the two tables.
"""

import argparse
import os
from typing import Optional, Tuple

import torch
import triton
from sgl_kernel import fp8_scaled_mm as sgl_scaled_mm

from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8
from sglang.utils import is_in_ci

IS_CI = is_in_ci()

# (N, K) shapes targeting each dispatch bucket boundary.
# Spans M16_smallN / M16_largeN / M32_largeN / M64_smallN / M64_largeN /
# M128_smallN / M128_largeN dispatch entries when crossed with batch sizes
# below.
NK_SHAPES = [
    (1024, 4096),
    (1024, 8192),
    (1280, 4096),  # n == kNThreshold boundary
    (4096, 4096),
    (4096, 8192),  # n == kM128NThreshold boundary for M128 bucket
    (8192, 4096),
    (8192, 8192),
    (14336, 4096),
    (14336, 8192),
    (28672, 4096),  # Llama-3 70B MLP up_proj N
    (28672, 8192),
]

# Batch sizes covering each M-bucket of the swap-AB dispatch.
# CI runs only M=1 to stay fast; full run probes the bucket transitions.
if IS_CI:
    batch_sizes = [1]
else:
    batch_sizes = [1, 8, 16, 17, 32, 48, 64, 96, 128]

line_vals = ["sglang-fp8-bf16", "sglang-fp8-fp16"]
line_names = line_vals
styles = [("blue", "-"), ("blue", "--")]


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_ = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    per_tensor_quant_fp8(input, output, scale, is_static)
    return output, scale


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=batch_sizes,
        x_log=False,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="GB/s",
        plot_name="fp8 swap-AB scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    dtype = torch.float16 if "fp16" in provider else torch.bfloat16

    a_fp8, scale_a_fp8 = sglang_scaled_fp8_quant(a, scale_a)
    b_fp8, scale_b_fp8 = sglang_scaled_fp8_quant(b, scale_b)
    b_fp8 = b_fp8.t()
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        lambda: sgl_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=None),
        quantiles=quantiles,
    )

    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save plots/CSVs (default: don't save)",
    )
    args = parser.parse_args()

    if IS_CI:
        # CI: probe a single (N, K) to stay quick.
        N, K = NK_SHAPES[0]
        print(f"N={N} K={K}: ")
        benchmark.run(print_data=True, N=N, K=K)
    else:
        for N, K in NK_SHAPES:
            print(f"N={N} K={K}: ")
            kwargs = {"print_data": True, "N": N, "K": K}
            if args.save_path:
                os.makedirs(args.save_path, exist_ok=True)
                kwargs["save_path"] = args.save_path
            benchmark.run(**kwargs)

    print("Benchmark finished!")
