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
from sglang.jit_kernel.kvcache import store_cache

_is_hip = bool(torch.version.hip)
HAS_AOT_STORE_CACHE = hasattr(torch.ops.sgl_kernel, "store_kv_cache")


def sglang_aot_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    set_kv_buffer_kernel(k_cache, v_cache, indices, k, v)


def sglang_jit_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    store_cache(k, v, k_cache, v_cache, indices)


@torch.compile()
def torch_compile_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    k_cache[indices] = k
    v_cache[indices] = v


alt_stream = torch.cuda.Stream()


def torch_streams_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    current_stream = torch.cuda.current_stream()
    alt_stream.wait_stream(current_stream)
    k_cache[indices] = k
    with torch.cuda.stream(alt_stream):
        v_cache[indices] = v
    current_stream.wait_stream(alt_stream)


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

LINE_VALS = ["jit", "torch_compile", "torch_streams"]
LINE_NAMES = ["SGL JIT Kernel", "PyTorch Compile", "PyTorch 2 Stream"]
STYLES = [("blue", "--"), ("red", ":"), ("green", "-.")]
# Keep non-HIP benchmark lines unchanged; only HIP tolerates missing AOT op.
if (not _is_hip) or HAS_AOT_STORE_CACHE:
    LINE_VALS = ["aot"] + LINE_VALS
    LINE_NAMES = ["SGL AOT Kernel"] + LINE_NAMES
    STYLES = [("orange", "-")] + STYLES
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
        plot_name="store-kvcache-performance",
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
        "jit": sglang_jit_store_cache,
        "torch_compile": torch_compile_store_cache,
        "torch_streams": torch_streams_store_cache,
    }
    if (not _is_hip) or HAS_AOT_STORE_CACHE:
        FN_MAP["aot"] = sglang_aot_store_cache

    def fn():
        impl = FN_MAP[provider]
        for i in range(NUM_LAYERS):
            impl(k[i], v[i], k_cache[i], v_cache[i], indices)

    # Custom time calculation: divide by NUM_LAYERS
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
