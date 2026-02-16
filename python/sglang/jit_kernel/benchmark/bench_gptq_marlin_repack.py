import os

import torch
import triton
import triton.testing
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack as jit_fn
from sglang.srt.layers.quantization.utils import gptq_quantize_weights, pack_rows

try:
    from sgl_kernel import gptq_marlin_repack as aot_fn

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Fixed problem dimensions
SIZE_N = 4096
NUM_BITS = 4
QUANT_TYPE = scalar_types.uint4b8
GROUP_SIZE = 128

# Pre-compute quantized weight for each size_k in the sweep
_cache = {}


def _get_inputs(size_k):
    if size_k not in _cache:
        size_n = SIZE_N
        b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")
        _, q_w, _, _, _ = gptq_quantize_weights(
            b_weight, QUANT_TYPE, GROUP_SIZE, act_order=False
        )
        q_w_gptq = pack_rows(q_w, NUM_BITS, size_k, size_n)
        sort_indices = torch.empty(0, dtype=torch.int, device="cuda")
        _cache[size_k] = (q_w_gptq, sort_indices)
    return _cache[size_k]


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    size_k = 4096
    q_w_gptq, sort_indices = _get_inputs(size_k)
    out_jit = jit_fn(q_w_gptq, sort_indices, size_k, SIZE_N, NUM_BITS)
    out_aot = aot_fn(q_w_gptq, sort_indices, size_k, SIZE_N, NUM_BITS)
    torch.testing.assert_close(out_jit, out_aot, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


if IS_CI:
    k_range = [128, 1024, 4096]
else:
    k_range = [128, 256, 512, 1024, 2048, 4096, 8192]

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
        x_names=["size_k"],
        x_vals=k_range,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="gptq-marlin-repack-performance",
        args={},
    )
)
def benchmark(size_k, provider):
    q_w_gptq, sort_indices = _get_inputs(size_k)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "jit":
        fn = lambda: jit_fn(q_w_gptq, sort_indices, size_k, SIZE_N, NUM_BITS)
    elif provider == "aot":
        fn = lambda: aot_fn(q_w_gptq, sort_indices, size_k, SIZE_N, NUM_BITS)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
