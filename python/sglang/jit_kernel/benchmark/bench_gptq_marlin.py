import os

import torch
import triton
import triton.testing
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm as jit_gptq_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.test.test_marlin_utils import marlin_quantize

try:
    from sgl_kernel import gptq_marlin_gemm as aot_gptq_marlin_gemm

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Fixed problem dimensions
SIZE_K = 4096
SIZE_N = 4096
GROUP_SIZE = 128
QUANT_TYPE = scalar_types.uint4b8

# Quantize weights once
_b_weight = torch.randn((SIZE_K, SIZE_N), dtype=torch.float16, device="cuda")
_w_ref, _marlin_q_w, _marlin_s, _g_idx, _sort_indices, _ = marlin_quantize(
    _b_weight, QUANT_TYPE, GROUP_SIZE, act_order=False
)
_workspace = marlin_make_workspace(_w_ref.device)


def _run_gemm(fn, a):
    return fn(
        a,
        None,
        _marlin_q_w,
        _marlin_s,
        None,
        None,
        _g_idx,
        _sort_indices,
        _workspace,
        QUANT_TYPE,
        a.shape[0],
        SIZE_N,
        SIZE_K,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    )


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    a = torch.randn((16, SIZE_K), dtype=torch.float16, device="cuda")
    out_jit = _run_gemm(jit_gptq_marlin_gemm, a)
    out_aot = _run_gemm(aot_gptq_marlin_gemm, a)
    torch.testing.assert_close(out_jit, out_aot, rtol=1e-3, atol=1e-3)
    print("Correctness check passed (JIT vs AOT)")


if IS_CI:
    m_range = [1, 16, 128]
else:
    m_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

if AOT_AVAILABLE:
    line_vals = ["jit", "aot"]
    line_names = ["JIT Kernel", "AOT Kernel"]
    styles = [("blue", "-"), ("green", "-")]
else:
    line_vals = ["jit"]
    line_names = ["JIT Kernel"]
    styles = [("blue", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size_m"],
        x_vals=m_range,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="gptq-marlin-gemm-performance",
        args={},
    )
)
def benchmark(size_m, provider):
    device = torch.device("cuda")
    a = torch.randn((size_m, SIZE_K), dtype=torch.float16, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "jit":
        fn = lambda: _run_gemm(jit_gptq_marlin_gemm, a)
    elif provider == "aot":
        fn = lambda: _run_gemm(aot_gptq_marlin_gemm, a)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
