"""Benchmark the set_mla_kv_buffer dispatcher.

Compares three providers across a batch-size sweep:
  - ``wrapper``:   the high-level wrapper exposed by ``set_mla_kv_buffer_triton``
                   (dispatches to TMA on SM90+, Triton fallback otherwise).
  - ``jit_tma``:   the JIT CUDA TMA bulk-store kernel directly.
  - ``triton``:    the BLOCK-tiled Triton kernel (SM<90 fallback path).
"""

import itertools
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.set_mla_kv_buffer import set_mla_kv_buffer as jit_set
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_kernel as sglang_triton_kernel
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton as sglang_wrapper
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, suite="stage-b-kernel-benchmark-1-gpu-large")


def _triton_baseline(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    sglang_triton_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        **pdl_kwargs,
    )


NUM_LAYERS = 8
CACHE_SIZE = (2 * 1024 * 1024) // NUM_LAYERS

NOPE_DIM = 512
ROPE_DIM = 64

BS_RANGE = get_benchmark_range(
    full_range=[1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384],
    ci_range=[1, 128, 2048, 4096, 8192],
)

LINE_VALS = ["wrapper", "jit_tma", "triton"]
LINE_NAMES = ["Wrapper (auto)", "JIT TMA bulk-store", "Triton (BLOCK=128 baseline)"]
STYLES = [("blue", "-"), ("green", "--"), ("red", "-.")]
X_NAMES = ["batch_size"]
CONFIGS = list(itertools.product(BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=X_NAMES,
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="set-mla-kv-buffer-performance",
        args={},
    )
)
def benchmark(batch_size: int, provider: str) -> Tuple[float, float, float]:
    cache_k_nope = torch.randn(
        (NUM_LAYERS, batch_size, 1, NOPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    cache_k_rope = torch.randn(
        (NUM_LAYERS, batch_size, 1, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    kv_buffer = torch.randn(
        (NUM_LAYERS, CACHE_SIZE, 1, NOPE_DIM + ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    loc = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE)[:batch_size]
    torch.cuda.synchronize()

    FN_MAP = {
        "wrapper": sglang_wrapper,
        "jit_tma": lambda buf, loc, n, r: jit_set(buf, loc, n, r),
        "triton": _triton_baseline,
    }

    def fn():
        impl = FN_MAP[provider]
        for i in range(NUM_LAYERS):
            impl(kv_buffer[i], loc, cache_k_nope[i], cache_k_rope[i])

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=DEFAULT_QUANTILES
    )
    return (
        1000 * ms / NUM_LAYERS,
        1000 * max_ms / NUM_LAYERS,
        1000 * min_ms / NUM_LAYERS,
    )


if __name__ == "__main__":
    benchmark.run(print_data=True)
