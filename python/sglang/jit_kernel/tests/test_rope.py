import time

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel import FusedSetKVBufferArg as FusedSetKVBufferArgKernel
from sgl_kernel import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace_kernel,
)

from sglang.jit_kernel.rope import FusedSetKVBufferArg as FusedSetKVBufferArgJit
from sglang.jit_kernel.rope import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace_jit,
)

DEVICE = "cuda"


@triton.jit
def burn_kernel(out_ptr, iters: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.full((), pid + 1, dtype=tl.uint32)

    a = tl.full((), 1664525, dtype=tl.uint32)
    c = tl.full((), 1013904223, dtype=tl.uint32)
    sh = tl.full((), 13, dtype=tl.uint32)

    for _ in range(iters):
        x = x * a + c
        x = x ^ (x >> sh)

    if pid == 0:
        tl.store(out_ptr, x)


def triton_burn(ms: float, grid=(256,)):
    iters = int(ms * 20000)
    out = torch.empty((), device="cuda", dtype=torch.uint32)
    burn_kernel[grid](out, iters=iters)
    return out


def create_cos_sin_cache(rotary_dim, max_position_embeddings, base, dtype):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )

    t = torch.arange(max_position_embeddings, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("seq_len", [1, 512])
@pytest.mark.parametrize("num_qo_heads", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 16])
@pytest.mark.parametrize("head_dim", [64, 512])
@pytest.mark.parametrize("rotary_dim", [64, 128])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("save_kv_cache", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rope(
    bs,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    interleave: bool,
    enable_pdl: bool,
    save_kv_cache: bool,
    dtype: torch.dtype,
) -> None:
    if head_dim < rotary_dim:
        pytest.skip(f"{head_dim=} < {rotary_dim=}")
    if not save_kv_cache and enable_pdl:
        pytest.skip(f"({save_kv_cache=}, {enable_pdl=}) is not allowed")

    q = torch.randn(bs * seq_len, num_qo_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(bs * seq_len, num_kv_heads * head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(bs * seq_len, num_kv_heads * head_dim, device=DEVICE, dtype=dtype)

    KV_POOL_SIZE = bs * seq_len * 2
    k_buffer = torch.zeros(
        KV_POOL_SIZE, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    v_buffer = torch.zeros(
        KV_POOL_SIZE, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    out_cache_loc = torch.randperm(KV_POOL_SIZE, dtype=torch.int64, device=DEVICE)[
        : bs * seq_len
    ].clone()

    pos_ids = torch.arange(seq_len, device=DEVICE).repeat(bs)

    max_seq_len = seq_len
    base = 10000
    cos_sin_cache = create_cos_sin_cache(rotary_dim, max_seq_len, base, dtype)

    q_jit = q.clone()
    k_jit = k.clone()
    v_jit = v.clone()
    k_buffer_jit = k_buffer.clone()
    v_buffer_jit = v_buffer.clone()
    out_cache_loc_jit = out_cache_loc.clone()
    fused_set_kv_buffer_arg_jit = FusedSetKVBufferArgJit(
        value=v_jit,
        k_buffer=k_buffer_jit.view(k_buffer_jit.shape[0], -1),
        v_buffer=v_buffer_jit.view(v_buffer_jit.shape[0], -1),
        k_scale=None,
        v_scale=None,
        cache_loc=out_cache_loc_jit,
    )

    q_kernel = q.clone()
    k_kernel = k.clone()
    v_kernel = v.clone()
    k_buffer_kernel = k_buffer.clone()
    v_buffer_kernel = v_buffer.clone()
    out_cache_loc_kernel = out_cache_loc.clone()
    fused_set_kv_buffer_arg_kernel = FusedSetKVBufferArgKernel(
        value=v_kernel,
        k_buffer=k_buffer_kernel.view(k_buffer_kernel.shape[0], -1),
        v_buffer=v_buffer_kernel.view(v_buffer_kernel.shape[0], -1),
        k_scale=None,
        v_scale=None,
        cache_loc=out_cache_loc_kernel,
    )

    stream_jit = torch.cuda.Stream()
    stream_kernel = torch.cuda.Stream()

    triton_burn(10, grid=(1024,))
    r = torch.randn_like(q)
    r_jit, r_kernel = r.clone(), r.clone()
    torch.cuda.synchronize()

    with torch.cuda.stream(stream_jit):
        # Test if rotary_embedding runs on stream_jit
        triton_burn(10, grid=(1024,))
        q_jit = q_jit + r_jit
        apply_rope_with_cos_sin_cache_inplace_jit(
            positions=pos_ids,
            query=q_jit,
            key=k_jit,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=(not interleave),
            fused_set_kv_buffer_arg=(
                fused_set_kv_buffer_arg_jit if save_kv_cache else None
            ),
            enable_pdl=enable_pdl,
        )

    with torch.cuda.stream(stream_kernel):
        triton_burn(10, grid=(1024,))
        q_kernel = q_kernel + r_kernel
        apply_rope_with_cos_sin_cache_inplace_kernel(
            positions=pos_ids,
            query=q_kernel,
            key=k_kernel,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=(not interleave),
            fused_set_kv_buffer_arg=(
                fused_set_kv_buffer_arg_kernel if save_kv_cache else None
            ),
            enable_pdl=enable_pdl,
        )

    torch.cuda.synchronize()

    atol = 1e-3 if dtype != torch.float32 else 1e-6
    rtol = 1e-3 if dtype != torch.float32 else 1e-6
    torch.testing.assert_close(q_jit, q_kernel, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_jit, k_kernel, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_buffer_jit, k_buffer_kernel, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_buffer_jit, v_buffer_kernel, atol=atol, rtol=rtol)


@pytest.mark.parametrize("bs", [8])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("num_qo_heads", [16])
@pytest.mark.parametrize("num_kv_heads", [16])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("rotary_dim", [64])
@pytest.mark.parametrize("interleave", [False])
@pytest.mark.parametrize("enable_pdl", [False])
@pytest.mark.parametrize("save_kv_cache", [False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_bench_rope(
    bs,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    interleave: bool,
    enable_pdl: bool,
    save_kv_cache: bool,
    dtype: torch.dtype,
) -> None:
    if head_dim < rotary_dim:
        pytest.skip(f"{head_dim=} < {rotary_dim=}")
    if not save_kv_cache and enable_pdl:
        pytest.skip(f"({save_kv_cache=}, {enable_pdl=}) is not allowed")

    q = torch.randn(bs * seq_len, num_qo_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(bs * seq_len, num_kv_heads * head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(bs * seq_len, num_kv_heads * head_dim, device=DEVICE, dtype=dtype)

    KV_POOL_SIZE = bs * seq_len * 2
    k_buffer = torch.zeros(
        KV_POOL_SIZE, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    v_buffer = torch.zeros(
        KV_POOL_SIZE, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    out_cache_loc = torch.randperm(KV_POOL_SIZE, dtype=torch.int64, device=DEVICE)[
        : bs * seq_len
    ].clone()

    pos_ids = torch.arange(seq_len, device=DEVICE).repeat(bs)

    max_seq_len = seq_len
    base = 10000
    cos_sin_cache = create_cos_sin_cache(rotary_dim, max_seq_len, base, dtype)

    q_jit = q.clone()
    k_jit = k.clone()
    v_jit = v.clone()
    k_buffer_jit = k_buffer.clone()
    v_buffer_jit = v_buffer.clone()
    out_cache_loc_jit = out_cache_loc.clone()

    q_kernel = q.clone()
    k_kernel = k.clone()
    v_kernel = v.clone()
    k_buffer_kernel = k_buffer.clone()
    v_buffer_kernel = v_buffer.clone()
    out_cache_loc_kernel = out_cache_loc.clone()

    jit_args = {
        "positions": pos_ids,
        "query": q_jit,
        "key": k_jit,
        "head_size": head_dim,
        "cos_sin_cache": cos_sin_cache,
        "is_neox": (not interleave),
        "fused_set_kv_buffer_arg": None,
        "enable_pdl": enable_pdl,
    }
    jit_time = bench_rope(
        apply_rope_with_cos_sin_cache_inplace_jit,
        jit_args,
    )
    kernel_args = {
        "positions": pos_ids,
        "query": q_kernel,
        "key": k_kernel,
        "head_size": head_dim,
        "cos_sin_cache": cos_sin_cache,
        "is_neox": (not interleave),
        "fused_set_kv_buffer_arg": None,
        "enable_pdl": enable_pdl,
    }
    kernel_time = bench_rope(
        apply_rope_with_cos_sin_cache_inplace_kernel,
        kernel_args,
    )
    print(f"\nPerformance Test - Batch={bs}, SeqLen={seq_len}")
    print(f"JIT: {jit_time*1000:.9f}ms, SGL: {kernel_time*1000:.9f}ms")
    if kernel_time > 0:
        speedup = kernel_time / jit_time if jit_time > 0 else float("inf")
        print(f"Speedup (SGL/JIT): {speedup:.2f}x")


def bench_rope(fn, args):
    warmup = 10
    iteration = 100
    for _ in range(warmup):
        fn(**args)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iteration):
        fn(**args)
    torch.cuda.synchronize()
    return (time.time() - start_time) / iteration


if __name__ == "__main__":
    pytest.main([__file__])
