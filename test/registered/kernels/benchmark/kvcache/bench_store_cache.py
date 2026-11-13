import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    create_empty,
    create_random,
)
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import _try_set_kv_buffer_fp8
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)
register_amd_ci(est_time=12, stage="jit-kernel-benchmark", runner_config="amd")


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


# Asymmetric K/V (head_dim != v_head_dim). The item_size sweep above drives both
# rows from one value, so it never reaches the split-prefix-plus-tail path.
# 192/128 and 384/256 both go live in a single MiMoV2 TP=4 deployment, whose
# layers carry either 1 or 2 kv heads per rank; the reversed and wide pairs cover
# a V-side tail and a num_split > 1 shape.
ASYM_ITEM_SIZES = [(192, 128), (384, 256), (128, 192), (1024, 512)]


@marker.parametrize("k_item,v_item", ASYM_ITEM_SIZES, [(192, 128), (1024, 512)])
@marker.parametrize("batch_size", [2**n for n in range(0, 15)], [16])
@marker.benchmark("impl", ["jit", "torch_compile", "torch_streams"])
def benchmark_asymmetric(batch_size: int, k_item: int, v_item: int, impl: str):
    torch.manual_seed(42)
    k = create_random(batch_size, k_item)
    k_cache = create_empty(CACHE_SIZE, k_item)
    v = create_random(batch_size, v_item)
    v_cache = create_empty(CACHE_SIZE, v_item)
    indices = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE)[:batch_size]
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(k, v, k_cache, v_cache, indices),
        graph_clone_args=(0, 1, 4),  # not need to clone cache, which is large
        memory_args=(k, v, indices),  # k_cache / v_cache excluded
        memory_output=(k, v),  # inplace write, size = k + v
    )


def eager_store_cache_fp8(k, v, k_cache, v_cache, indices, k_scale, v_scale):
    # The pool's eager writer: divide, cast and store, five launches with scales.
    if k_scale is not None:
        k, v = k / k_scale, v / v_scale
    k = k.to(fp8_dtype).view(torch.uint8)
    v = v.to(fp8_dtype).view(torch.uint8)
    store_cache(k, v, k_cache.view(torch.uint8), v_cache.view(torch.uint8), indices)


def fused_store_cache_fp8(k, v, k_cache, v_cache, indices, k_scale, v_scale):
    # False would mean nothing ran (e.g. below SM89), so a timing would be meaningless.
    assert _try_set_kv_buffer_fp8(
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        indices=indices,
        k_scale=k_scale,
        v_scale=v_scale,
        row_dim=k.shape[-1],
        size_limit=k_cache.shape[0],
    )


FP8_FN_MAP = {"eager": eager_store_cache_fp8, "fused": fused_store_cache_fp8}


@marker.parametrize("scales", ["tensor", "none"], ["tensor"])
@marker.parametrize("item_size", [128, 256, 1024, 2048], [1024])
@marker.parametrize("batch_size", [2**n for n in range(0, 15)], [16, 4096])
@marker.benchmark("impl", ["eager", "fused"])
def benchmark_fp8(batch_size: int, item_size: int, scales: str, impl: str):
    torch.manual_seed(42)
    k = create_random(batch_size, item_size)
    v = create_random(batch_size, item_size)
    k_cache = create_empty(CACHE_SIZE, item_size, dtype=fp8_dtype)
    v_cache = create_empty(CACHE_SIZE, item_size, dtype=fp8_dtype)
    # Slot 0 is the padding slot both writers skip.
    indices = torch.randperm(CACHE_SIZE - 1, device=DEFAULT_DEVICE)[:batch_size] + 1
    scale = torch.tensor(0.5, device=DEFAULT_DEVICE) if scales == "tensor" else None
    return marker.do_bench(
        FP8_FN_MAP[impl],
        input_args=(k, v, k_cache, v_cache, indices, scale, scale),
        graph_clone_args=(0, 1, 4),
        memory_args=(k, v, indices),
        memory_output=(k_cache[:batch_size], v_cache[:batch_size]),
    )


if __name__ == "__main__":
    benchmark.run()
    benchmark_asymmetric.run()
    benchmark_fp8.run()
