import itertools
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import set_kv_buffer_kernel

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.store import store_kv_cache


def sglang_aot_store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    set_kv_buffer_kernel(k_cache, v_cache, indices, k, v)


def sglang_jit_store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    store_kv_cache(k_cache, v_cache, indices, k, v)


NUM_LAYERS = 8
CACHE_SIZE = 2 * 1024 * 1024 // NUM_LAYERS

BS_RANGE = get_benchmark_range(
    full_range=[2**n for n in range(0, 15)],
    ci_range=[16],
)
ITEM_SIZE = get_benchmark_range(
    full_range=[64, 128, 256, 512, 1024],
    ci_range=[1024],
)

LINE_VALS = ["aot", "jit"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel"]
STYLES = [("orange", "-"), ("blue", "--")]
X_NAMES = ["item_size", "batch_size"]
CONFIGS = list(itertools.product(ITEM_SIZE, BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=X_NAMES,
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="store-kv-cache-performance",
        args={},
    )
)
def benchmark(
    batch_size: int, item_size: int, provider: str
) -> Tuple[float, float, float]:
    k = torch.randn(
        (NUM_LAYERS, batch_size, item_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    v = torch.randn(
        (NUM_LAYERS, batch_size, item_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    k_cache = torch.randn(
        (NUM_LAYERS, CACHE_SIZE, item_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    v_cache = torch.randn(
        (NUM_LAYERS, CACHE_SIZE, item_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    indices = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE)[:batch_size]
    torch.cuda.synchronize()

    FN_MAP = {
        "aot": sglang_aot_store_kv_cache,
        "jit": sglang_jit_store_kv_cache,
    }

    def fn():
        impl = FN_MAP[provider]
        for i in range(NUM_LAYERS):
            impl(k[i], v[i], k_cache[i], v_cache[i], indices)

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
