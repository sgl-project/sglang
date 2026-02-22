import os

import numpy as np
import torch
import triton
import triton.testing
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.awq_marlin_repack import (
    awq_marlin_repack as jit_awq_marlin_repack,
)
from sglang.srt.layers.quantization.utils import pack_cols, quantize_weights

try:
    from sgl_kernel import awq_marlin_repack as aot_awq_marlin_repack

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
NUM_BITS = 4
GROUP_SIZE = 128


def awq_pack(q_w, num_bits, size_k, size_n):
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_w = q_w.reshape((-1, len(interleave)))[:, interleave].ravel()
    q_w = q_w.reshape((-1, size_n)).contiguous()
    return pack_cols(q_w, num_bits, size_k, size_n)


# Quantize weights once
_b_weight = torch.randn((SIZE_K, SIZE_N), dtype=torch.float16, device="cuda")
_w_ref, _q_w, _s, _zp = quantize_weights(
    _b_weight, scalar_types.uint4, GROUP_SIZE, zero_points=True
)
_q_w_awq = awq_pack(_q_w, NUM_BITS, SIZE_K, SIZE_N)


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return
    out_jit = jit_awq_marlin_repack(_q_w_awq, SIZE_K, SIZE_N, NUM_BITS)
    out_aot = aot_awq_marlin_repack(_q_w_awq, SIZE_K, SIZE_N, NUM_BITS)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_jit, out_aot, rtol=0, atol=0)
    print("Correctness check passed (JIT vs AOT)")


if IS_CI:
    k_range = [1024, 4096]
else:
    k_range = [512, 1024, 2048, 4096, 8192]

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
        plot_name="awq-marlin-repack-performance",
        args={"size_n": SIZE_N, "num_bits": NUM_BITS},
    )
)
def benchmark(size_k, size_n, num_bits, provider):
    group_size = min(GROUP_SIZE, size_k)

    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")
    w_ref, q_w, s, zp = quantize_weights(
        b_weight, scalar_types.uint4, group_size, zero_points=True
    )
    q_w_awq = awq_pack(q_w, num_bits, size_k, size_n)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "jit":
        fn = lambda: jit_awq_marlin_repack(q_w_awq, size_k, size_n, num_bits)
    elif provider == "aot":
        fn = lambda: aot_awq_marlin_repack(q_w_awq, size_k, size_n, num_bits)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
