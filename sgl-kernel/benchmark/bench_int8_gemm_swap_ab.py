"""Targeted benchmark for the SM90 int8 swap-AB dispatch path (PR-B).

Sweeps small batch sizes (M = 1..128) across (N, K) shapes that exercise each
dispatch bucket in int8_gemm_sm90_dispatch.cuh, comparing sgl-kernel int8
scaled_mm against vLLM's cutlass_scaled_mm. Output mirrors
bench_fp8_gemm_swap_ab.py (triton.testing.perf_report, GB/s).

Two-build A/B (the swap-AB delta over the current int8 SM90 path):
  1. on `main`:           python bench_int8_gemm_swap_ab.py > main.txt
  2. on feature branch:   python bench_int8_gemm_swap_ab.py > swap.txt
  3. diff the "sglang-*" columns of main.txt vs swap.txt
"""

import argparse
import os

import torch
import triton
from sgl_kernel import int8_scaled_mm as sgl_scaled_mm

from sglang.utils import is_in_ci

# Optional vLLM baseline (cross-engine anchor).
try:
    from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm

    VLLM_AVAILABLE = True
except ImportError:
    vllm_scaled_mm = None
    VLLM_AVAILABLE = False

IS_CI = is_in_ci()

# (N, K) shapes hitting each dispatch bucket boundary. All satisfy the int8
# kernel constraints: K % 16 == 0 and N % 8 == 0 (validated below).
NK_SHAPES = [
    (1024, 4096),
    (1280, 4096),  # n == kNThreshold boundary
    (4096, 4096),
    (4096, 8192),  # n == kM128NThreshold boundary
    (8192, 8192),
    (14336, 4096),
    (28672, 8192),  # Llama-3-70B MLP up_proj N
]
for _n, _k in NK_SHAPES:
    assert _k % 16 == 0, f"K={_k} must be a multiple of 16 (int8 alignment)"
    assert _n % 8 == 0, f"N={_n} must be a multiple of 8 (int8 alignment)"

# Small-M focus: swap-AB only acts for M <= 64; include 96/128 to show the
# crossover into the non-swap path. CI runs M=1 only to stay fast.
if IS_CI:
    batch_sizes = [1]
else:
    batch_sizes = [1, 8, 16, 17, 32, 48, 64, 96, 128]

if VLLM_AVAILABLE:
    line_vals = ["sglang-bf16", "sglang-fp16", "vllm-bf16", "vllm-fp16"]
    styles = [("blue", "-"), ("blue", "--"), ("green", "-"), ("green", "--")]
else:
    line_vals = ["sglang-bf16", "sglang-fp16"]
    styles = [("blue", "-"), ("blue", "--")]
line_names = line_vals


def to_int8(t: torch.Tensor) -> torch.Tensor:
    return torch.round(t.clamp(min=-128, max=127)).to(dtype=torch.int8)


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
        plot_name="int8 swap-AB scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    dtype = torch.float16 if provider.endswith("fp16") else torch.bfloat16
    a = to_int8(torch.randn((M, K), device="cuda") * 5)
    b = to_int8(torch.randn((N, K), device="cuda").t() * 5)
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    # bias dtype must match the output dtype (kernel TORCH_CHECK).
    bias = torch.randn((N,), device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider.startswith("vllm"):
        if not VLLM_AVAILABLE:
            return (0, 0, 0)
        fn = lambda: vllm_scaled_mm(a, b, scale_a, scale_b, dtype, bias)
    else:
        fn = lambda: sgl_scaled_mm(a, b, scale_a, scale_b, dtype, bias)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    # Same throughput proxy as bench_int8_gemm.py for cross-consistency.
    gbps = (
        lambda ms: (
            (2 * M * N * K - M * N) * a.element_size()
            + (3 * M * N) * scale_a.element_size()
        )
        * 1e-9
        / (ms * 1e-3)
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default=None, help="Directory to save plots/CSVs"
    )
    args = parser.parse_args()

    shapes = NK_SHAPES[:1] if IS_CI else NK_SHAPES
    for N, K in shapes:
        print(f"N={N} K={K}:")
        kwargs = {"print_data": True, "N": N, "K": K}
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            kwargs["save_path"] = args.save_path
        benchmark.run(**kwargs)

    print("Benchmark finished!")
