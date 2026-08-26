"""Benchmark the set_mla_kv_buffer dispatcher.

Compares three providers across a batch-size sweep:
  - ``wrapper``:   the high-level wrapper exposed by ``set_mla_kv_buffer_triton``
                   (dispatches to TMA on SM90+, Triton fallback otherwise).
  - ``jit_tma``:   the JIT CUDA TMA bulk-store kernel directly.
  - ``triton``:    the BLOCK-tiled Triton kernel (SM<90 fallback path).
"""

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
)
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer as jit_set
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_kernel as sglang_triton_kernel
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton as sglang_wrapper
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=9, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


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
        0,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,  # type: ignore
        DCP_RANK=0,  # type: ignore
        DCP_WORLD_SIZE=1,  # type: ignore
        **pdl_kwargs,  # type: ignore
    )


# 2M elements
CACHE_SIZE = 2 * 1024 * 1024
NOPE_DIM = 512
ROPE_DIM = 64


@marker.parametrize("batch_size", marker.range(15, pattern="pow2"), [1, 128, 8192])
@marker.benchmark("provider", ["wrapper", "jit_tma", "triton"])
def benchmark(batch_size: int, provider: str):
    cache_k_nope = torch.randn(
        (batch_size, 1, NOPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    cache_k_rope = torch.randn(
        (batch_size, 1, ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, NOPE_DIM + ROPE_DIM),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    loc = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE)[:batch_size]

    FN_MAP = {
        "wrapper": sglang_wrapper,
        "jit_tma": lambda buf, loc, n, r: jit_set(buf, loc, n, r),
        "triton": _triton_baseline,
    }

    return marker.do_bench(
        FN_MAP[provider],
        input_args=(kv_buffer, loc, cache_k_nope, cache_k_rope),
        graph_clone_args=(1, 2, 3),
        memory_args=(loc, cache_k_nope, cache_k_rope),
        memory_output=(cache_k_nope, cache_k_rope),
    )


if __name__ == "__main__":
    benchmark.run()
