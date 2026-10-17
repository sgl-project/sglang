import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE, create_random
from sglang.kernels.ops.kvcache.cache_move import store_cache_fp8
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FP8 = torch.float8_e4m3fn
CACHE_SIZE = 1024 * 1024


def eager_store(k, v, k_cache, v_cache, indices, k_scale, v_scale):
    # The pool's eager FP8 writer: divide, cast, then the byte-copy store.
    if k_scale is not None:
        k = k / k_scale
    if v_scale is not None:
        v = v / v_scale
    row_dim = k_cache.shape[-2] * k_cache.shape[-1]
    store_cache(
        k.to(FP8).view(torch.uint8).view(-1, row_dim),
        v.to(FP8).view(torch.uint8).view(-1, row_dim),
        k_cache.view(torch.uint8).view(-1, row_dim),
        v_cache.view(torch.uint8).view(-1, row_dim),
        indices,
    )


def fused_store(k, v, k_cache, v_cache, indices, k_scale, v_scale):
    store_cache_fp8(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        indices=indices,
        k_scale=k_scale,
        v_scale=v_scale,
    )


FN_MAP = {"eager": eager_store, "fused": fused_store}


@marker.parametrize("scale", ["tensor", "none"], ["tensor"])
@marker.parametrize(
    "num_kv_heads,head_dim",
    [(1, 128), (2, 128), (8, 128), (8, 64), (4, 256)],
    [(8, 128)],
)
@marker.parametrize("batch_size", [2**n for n in range(0, 15)], [16, 4096])
@marker.benchmark("impl", ["eager", "fused"])
def benchmark(batch_size: int, num_kv_heads: int, head_dim: int, scale: str, impl: str):
    torch.manual_seed(42)
    k = create_random(batch_size, num_kv_heads, head_dim)
    v = create_random(batch_size, num_kv_heads, head_dim)
    k_cache = torch.empty(
        (CACHE_SIZE, num_kv_heads, head_dim), dtype=FP8, device=DEFAULT_DEVICE
    )
    v_cache = torch.empty_like(k_cache)
    # Slot zero is the reserved padding slot that both writers skip.
    indices = torch.randperm(CACHE_SIZE - 1, device=DEFAULT_DEVICE)[:batch_size] + 1
    k_scale = v_scale = None
    if scale == "tensor":
        k_scale = torch.tensor(0.5, dtype=torch.float32, device=DEFAULT_DEVICE)
        v_scale = torch.tensor(0.7, dtype=torch.float32, device=DEFAULT_DEVICE)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(k, v, k_cache, v_cache, indices, k_scale, v_scale),
        graph_clone_args=(0, 1, 4),  # caches are large; clone only the read args
        memory_args=(k, v, indices),
        memory_output=None,
        extra_memory_footprint=k.numel() + v.numel(),  # one FP8 byte per element
    )


if __name__ == "__main__":
    benchmark.run()
