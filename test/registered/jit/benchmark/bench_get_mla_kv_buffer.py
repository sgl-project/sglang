"""Benchmark the get_mla_kv_buffer dispatcher for Kimi K2.5 MLA rows."""

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    create_empty,
    create_random,
)
from sglang.srt.mem_cache.utils import get_mla_kv_buffer_triton as sglang_wrapper
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-benchmark-1-gpu-large")

CACHE_SIZE = 100_000

NOPE_DIM = 512
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM


@triton.jit
def _old_one_cta_get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    loc_src_ptr = kv_buffer_ptr + loc * buffer_stride

    nope_offs = tl.arange(0, nope_dim)
    nope_src = tl.load(loc_src_ptr + nope_offs)
    tl.store(cache_k_nope_ptr + pid_loc * nope_stride + nope_offs, nope_src)

    rope_offs = tl.arange(0, rope_dim)
    rope_src = tl.load(loc_src_ptr + nope_dim + rope_offs)
    tl.store(cache_k_rope_ptr + pid_loc * rope_stride + rope_offs, rope_src)


def _old_one_cta_get_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    grid = (loc.numel(),)
    _old_one_cta_get_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


@torch.compile()
def _torch_get_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    cache_k_nope.copy_(kv_buffer[loc, :, :NOPE_DIM])
    cache_k_rope.copy_(kv_buffer[loc, :, NOPE_DIM:TOTAL_DIM])


FN_MAP = {
    "wrapper": sglang_wrapper,
    "old_one_cta": _old_one_cta_get_mla_kv_buffer,
    "torch_compile": _torch_get_mla_kv_buffer,
}


@marker.parametrize(
    "batch_size",
    [
        1,
        4,
        32,
        128,
        255,
        256,
        511,
        512,
        2048,
        4096,
        16384,
        32768,
        65536,
        73728,
        74240,
        81920,
    ],
    [128, 255, 256, 4096, 74240],
)
@marker.benchmark("impl", ["wrapper", "old_one_cta", "torch_compile"])
def benchmark(batch_size: int, impl: str):
    torch.manual_seed(42)
    kv_buffer = create_random(CACHE_SIZE, 1, TOTAL_DIM)
    loc = torch.randperm(CACHE_SIZE, device=DEFAULT_DEVICE, dtype=torch.int64)[
        :batch_size
    ]
    cache_k_nope = create_empty(batch_size, 1, NOPE_DIM)
    cache_k_rope = create_empty(batch_size, 1, ROPE_DIM)

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(kv_buffer, loc, cache_k_nope, cache_k_rope),
        graph_clone_args=(0, 1),
        memory_args=(kv_buffer, loc),
        memory_output=(cache_k_nope, cache_k_rope),
    )


if __name__ == "__main__":
    benchmark.run()
