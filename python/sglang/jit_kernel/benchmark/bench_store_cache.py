import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    create_empty,
    create_random,
)
from sglang.jit_kernel.kvcache import store_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, suite="base-b-kernel-benchmark-1-gpu-large")


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


CACHE_SIZE = 2 * 1024 * 1024
FN_MAP = {
    "jit": store_cache,
    "torch_compile": torch_compile_store_cache,
    "torch_streams": torch_streams_store_cache,
}


@marker.parametrize("item_size", [64, 128, 256, 512, 1024], [1024])
@marker.parametrize("batch_size", [2**n for n in range(0, 15)], [16])
@marker.benchmark("impl", ["jit", "torch_compile", "torch_streams"])
def benchmark(batch_size: int, item_size: int, impl: str):
    torch.manual_seed(42)
    k = create_random(batch_size, item_size)
    k_cache = create_empty(CACHE_SIZE, item_size)
    v = create_random(batch_size, item_size)
    v_cache = create_empty(CACHE_SIZE, item_size)
    indices = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE)[:batch_size]
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(k, v, k_cache, v_cache, indices),
        graph_clone_args=(0, 1, 4),  # not need to clone cache, which is large
        memory_args=(k, v, indices),  # k_cache / v_cache excluded
        memory_output=(k, v),  # inplace write, size = k + v
    )


if __name__ == "__main__":
    benchmark.run()
