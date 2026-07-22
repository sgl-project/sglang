import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, create_random
from sglang.jit_kernel.kvcache import store_cache, store_cache_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=9, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

QUANT_DST = torch.float8_e4m3fn


def eager_quant_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    # The unfused MHATokenToKVPool.set_kv_buffer sequence this kernel replaces:
    # in-place div + dtype cast per tensor, then the byte store.
    k.div_(k_scale)
    v.div_(v_scale)
    k8 = k.to(QUANT_DST)
    v8 = v.to(QUANT_DST)
    store_cache(
        k8.view(torch.uint8),
        v8.view(torch.uint8),
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        indices,
    )


def fused_quant_store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    store_cache_quant(k, v, k_cache, v_cache, indices, k_scale, v_scale)


CACHE_SIZE = 2 * 1024 * 1024
FN_MAP = {
    "jit_fused": fused_quant_store_cache,
    "eager": eager_quant_store_cache,
}


@marker.parametrize("item_size", [64, 128, 256, 512, 1024], [1024])
@marker.parametrize("batch_size", [2**n for n in range(0, 15)], [16])
@marker.benchmark("impl", ["jit_fused", "eager"])
def benchmark(batch_size: int, item_size: int, impl: str):
    torch.manual_seed(42)
    k = create_random(batch_size, item_size)
    v = create_random(batch_size, item_size)
    k_cache = torch.empty(
        (CACHE_SIZE, item_size), dtype=QUANT_DST, device=DEFAULT_DEVICE
    )
    v_cache = torch.empty(
        (CACHE_SIZE, item_size), dtype=QUANT_DST, device=DEFAULT_DEVICE
    )
    indices = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE)[:batch_size]
    k_scale = torch.tensor([1.7], dtype=torch.float32, device=DEFAULT_DEVICE)
    v_scale = torch.tensor([0.9], dtype=torch.float32, device=DEFAULT_DEVICE)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(k, v, k_cache, v_cache, indices, k_scale, v_scale),
        # k / v are read (and mutated by the eager path) -> clone per iter;
        # the caches are large enough not to stay L2-hot.
        graph_clone_args=(0, 1, 4),
        memory_args=(k, v, indices),  # k_cache / v_cache excluded
        memory_output=(k_cache[:batch_size], v_cache[:batch_size]),  # fp8 rows written
    )


if __name__ == "__main__":
    benchmark.run()
