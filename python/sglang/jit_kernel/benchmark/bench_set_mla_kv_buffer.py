"""Compare the JIT set_mla_kv_buffer kernel against:
  1. The current SGLang Triton kernel (set_mla_kv_buffer_kernel).
  2. The lightseek/tokenspeed reference kernel (block-split + per-loc dispatch),
     reproduced inline from
     https://github.com/lightseekorg/tokenspeed/blob/ea9799067810479da9e7473fcbee48eb900de65e/python/tokenspeed/runtime/cache/utils.py
"""

import itertools
from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.set_mla_kv_buffer import set_mla_kv_buffer as jit_set
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.mem_cache.utils import (
    set_mla_kv_buffer_kernel as sglang_triton_kernel,
    set_mla_kv_buffer_triton as sglang_hybrid,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, suite="stage-b-kernel-benchmark-1-gpu-large")


# ----------------- SGLang Triton (current production) -----------------


def _triton_sglang(kv_buffer, loc, cache_k_nope, cache_k_rope):
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


# ----------------- tokenspeed reference (lightseek) -----------------
# Reproduced verbatim from the URL above (omitting comments / license header).


@triton.jit
def _ts_block_split_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)
    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim
    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs
    if base + BLOCK <= nope_dim:
        src = tl.load(cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask)
    else:
        offs_rope = offs - nope_dim
        src = tl.load(cache_k_rope_ptr + pid_loc * rope_stride + offs_rope, mask=mask)
    tl.store(dst_ptr, src, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ts_per_loc_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    n_loc,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_LOC: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    loc_indices = pid * BLOCK_LOC + tl.arange(0, BLOCK_LOC)
    loc_mask = loc_indices < n_loc
    locs = tl.load(loc_ptr + loc_indices, mask=loc_mask, other=0)
    nope_offs = tl.arange(0, nope_dim)
    src_nope = tl.load(
        cache_k_nope_ptr + loc_indices[:, None] * nope_stride + nope_offs[None, :],
        mask=loc_mask[:, None],
    )
    tl.store(
        kv_buffer_ptr + locs[:, None] * buffer_stride + nope_offs[None, :],
        src_nope,
        mask=loc_mask[:, None],
    )
    rope_offs = tl.arange(0, rope_dim)
    src_rope = tl.load(
        cache_k_rope_ptr + loc_indices[:, None] * rope_stride + rope_offs[None, :],
        mask=loc_mask[:, None],
    )
    tl.store(
        kv_buffer_ptr + locs[:, None] * buffer_stride + nope_dim + rope_offs[None, :],
        src_rope,
        mask=loc_mask[:, None],
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _triton_tokenspeed(kv_buffer, loc, cache_k_nope, cache_k_rope):
    n_loc = loc.numel()
    nope_dim = cache_k_nope.size(-1)
    rope_dim = cache_k_rope.size(-1)
    enable_pdl = is_arch_support_pdl()
    pdl_kw = {"launch_pdl": True} if enable_pdl else {}
    if n_loc >= 512:
        if n_loc >= 16384:
            block_loc, num_warps, num_stages = 4, 1, 2
        elif n_loc >= 2048:
            block_loc, num_warps, num_stages = 4, 4, 2
        else:
            block_loc, num_warps, num_stages = 2, 4, 2
        grid = (triton.cdiv(n_loc, block_loc),)
        _ts_per_loc_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            n_loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK_LOC=block_loc,
            ENABLE_PDL=enable_pdl,
            num_warps=num_warps,
            num_stages=num_stages,
            **pdl_kw,
        )
    else:
        BLOCK = 256
        grid = (n_loc, triton.cdiv(nope_dim + rope_dim, BLOCK))
        _ts_block_split_kernel[grid](
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
            ENABLE_PDL=enable_pdl,
            **pdl_kw,
        )


# ----------------- Benchmark scaffolding -----------------

NUM_LAYERS = 8
CACHE_SIZE = (2 * 1024 * 1024) // NUM_LAYERS

NOPE_DIM = 512
ROPE_DIM = 64

BS_RANGE = get_benchmark_range(
    full_range=[1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384],
    ci_range=[1, 128, 2048, 4096, 8192],
)

LINE_VALS = ["hybrid", "jit", "triton_sgl", "triton_ts"]
LINE_NAMES = [
    "Hybrid (Triton+JIT)",
    "JIT (auto)",
    "Triton (SGL)",
    "Triton (tokenspeed)",
]
STYLES = [("blue", "-"), ("green", "--"), ("red", "-."), ("purple", "-.")]
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
        "hybrid": sglang_hybrid,
        "jit": lambda buf, loc, n, r: jit_set(buf, loc, n, r),
        "triton_sgl": _triton_sglang,
        "triton_ts": _triton_tokenspeed,
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
