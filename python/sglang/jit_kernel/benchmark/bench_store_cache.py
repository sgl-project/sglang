import itertools
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import set_kv_buffer_kernel

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.kvcache import store_cache

IS_CI = is_in_ci()


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


DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_LAYERS = 8
CACHE_SIZE = 2 * 1024 * 1024 // NUM_LAYERS

if IS_CI:
    BS_RANGE = [16]
    ITEM_SIZE = [1024]
else:
    BS_RANGE = [2**n for n in range(0, 15)]
    ITEM_SIZE = [64, 128, 256, 512, 1024]

LINE_VALS = ["aot", "jit", "torch_compile", "torch_streams"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "PyTorch Compile", "PyTorch 2 Stream"]
STYLES = [("orange", "-"), ("blue", "--"), ("red", ":"), ("green", "-.")]
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
    k = torch.randn((NUM_LAYERS, batch_size, item_size), dtype=DTYPE, device=DEVICE)
    v = torch.randn((NUM_LAYERS, batch_size, item_size), dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(
        (NUM_LAYERS, CACHE_SIZE, item_size), dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.randn(
        (NUM_LAYERS, CACHE_SIZE, item_size), dtype=DTYPE, device=DEVICE
    )
    indices = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size]
    torch.cuda.synchronize()

    FN_MAP = {
        "aot": sglang_aot_store_cache,
        "jit": sglang_jit_store_cache,
        "torch_compile": torch_compile_store_cache,
        "torch_streams": torch_streams_store_cache,
    }

    def fn():
        impl = FN_MAP[provider]
        for i in range(NUM_LAYERS):
            impl(k[i], v[i], k_cache[i], v_cache[i], indices)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return (
        1000 * ms / NUM_LAYERS,
        1000 * max_ms / NUM_LAYERS,
        1000 * min_ms / NUM_LAYERS,
    )


if __name__ == "__main__":
    benchmark.run(print_data=True)
