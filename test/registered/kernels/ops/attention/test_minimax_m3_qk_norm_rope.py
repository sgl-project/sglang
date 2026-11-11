# SPDX-License-Identifier: Apache-2.0
"""Reference tests for MiniMax-M3 fused Q/K Gemma RMSNorm + RoPE."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils import is_hip

if not is_hip():
    pytest.skip(
        "MiniMax-M3 fused Q/K norm + RoPE kernel is ROCm-only.",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.attention.minimax_m3_qk_norm_rope import (  # noqa: E402
    qk_gemma_rmsnorm_rope,
    sparse_qk_index_gemma_rmsnorm_rope,
    sparse_qk_index_gemma_rmsnorm_rope_cache,
)
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

# ROCm-only fused kernel; the MI35x suite also covers its gfx950 tiled path.
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = "cuda"
EPS = 1e-6


def _gemma_norm_by_head(x: torch.Tensor, weight: torch.Tensor, head_dim: int):
    orig_shape = x.shape
    orig_dtype = x.dtype
    xh = x.view(x.shape[0], -1, head_dim).float()
    var = xh.pow(2).mean(dim=-1, keepdim=True)
    out = xh * torch.rsqrt(var + EPS) * (1.0 + weight.float())
    return out.to(orig_dtype).reshape(orig_shape)


def _apply_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
):
    orig_shape = x.shape
    xh = x.view(x.shape[0], -1, head_dim)
    x_rot = xh[..., :rotary_dim].float()
    x_pass = xh[..., rotary_dim:]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()

    if is_neox_style:
        x1, x2 = x_rot.chunk(2, dim=-1)
        y_rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        y_rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        y_rot = y_rot.flatten(-2)

    return torch.cat((y_rot.to(x.dtype), x_pass), dim=-1).reshape(orig_shape)


def _reference(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    cos_sin_cache,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    q_norm = _gemma_norm_by_head(q, q_weight, head_dim)
    k_norm = _gemma_norm_by_head(k, k_weight, head_dim)
    q_ref = _apply_rope_ref(
        q_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    k_ref = _apply_rope_ref(
        k_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    return q_ref, k_ref


def _sparse_reference(
    q,
    k,
    idx_q,
    idx_k,
    q_weight,
    k_weight,
    idx_q_weight,
    idx_k_weight,
    positions,
    cos_sin_cache,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    q_ref, k_ref = _reference(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    idx_q_norm = _gemma_norm_by_head(idx_q, idx_q_weight, head_dim)
    idx_k_norm = _gemma_norm_by_head(idx_k, idx_k_weight, head_dim)
    idx_q_ref = _apply_rope_ref(
        idx_q_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    idx_k_ref = _apply_rope_ref(
        idx_k_norm, positions, cos_sin_cache, head_dim, rotary_dim, is_neox_style
    )
    return q_ref, k_ref, idx_q_ref, idx_k_ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize(
    "num_tokens,q_heads,k_heads,head_dim,rotary_dim",
    [(1, 16, 1, 128, 64), (17, 16, 1, 128, 64), (64, 4, 1, 128, 64)],
)
@torch.inference_mode()
def test_qk_gemma_rmsnorm_rope_matches_reference(
    dtype, is_neox_style, num_tokens, q_heads, k_heads, head_dim, rotary_dim
):
    torch.manual_seed(0)
    q_dim = q_heads * head_dim
    k_dim = k_heads * head_dim
    padding_dim = 37
    qkv = torch.randn(
        num_tokens, q_dim + k_dim + padding_dim, device=DEVICE, dtype=dtype
    )
    q, k, _ = qkv.split([q_dim, k_dim, padding_dim], dim=-1)
    if num_tokens > 1:
        assert not q.is_contiguous()
        assert not k.is_contiguous()

    q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE, dtype=torch.long)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)

    got_q, got_k = qk_gemma_rmsnorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        EPS,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    ref_q, ref_k = _reference(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )

    torch.testing.assert_close(got_q, ref_q, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_k, ref_k, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize(
    "num_tokens,q_heads,k_heads,idx_q_heads,head_dim,rotary_dim",
    [(1, 16, 1, 16, 128, 64), (19, 16, 1, 16, 128, 64)],
)
@torch.inference_mode()
def test_sparse_qk_index_gemma_rmsnorm_rope_matches_reference(
    dtype,
    is_neox_style,
    num_tokens,
    q_heads,
    k_heads,
    idx_q_heads,
    head_dim,
    rotary_dim,
):
    torch.manual_seed(1)
    q = torch.randn(num_tokens, q_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_q = torch.randn(num_tokens, idx_q_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_k = torch.randn(num_tokens, head_dim, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE, dtype=torch.long)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)

    got = sparse_qk_index_gemma_rmsnorm_rope(
        q,
        k,
        idx_q,
        idx_k,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        EPS,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    ref = _sparse_reference(
        q,
        k,
        idx_q,
        idx_k,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    for got_tensor, ref_tensor in zip(got, ref):
        torch.testing.assert_close(got_tensor, ref_tensor, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox_style", [True, False])
@torch.inference_mode()
def test_sparse_qk_index_gemma_rmsnorm_rope_cache_matches_reference(
    dtype, is_neox_style
):
    torch.manual_seed(2)
    num_tokens, q_heads, k_heads, idx_q_heads = 11, 16, 1, 16
    head_dim, rotary_dim = 128, 64
    q = torch.randn(num_tokens, q_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_q = torch.randn(num_tokens, idx_q_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_k = torch.randn(num_tokens, head_dim, device=DEVICE, dtype=dtype)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_q_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    idx_k_weight = torch.randn(head_dim, device=DEVICE, dtype=torch.float32)
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE, dtype=torch.long)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)
    # Slot 0 is the reserved padding slot, which the main K/V store skips.
    out_cache_loc = torch.randperm(63, device=DEVICE)[:num_tokens] + 1
    k_cache = torch.empty(64, k_heads, head_dim, device=DEVICE, dtype=dtype)
    v_cache = torch.empty(64, k_heads, head_dim, device=DEVICE, dtype=dtype)
    idx_k_cache = torch.empty(64, 1, head_dim, device=DEVICE, dtype=dtype)

    got = sparse_qk_index_gemma_rmsnorm_rope_cache(
        q,
        k,
        v,
        idx_q,
        idx_k,
        k_cache,
        v_cache,
        idx_k_cache,
        out_cache_loc,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        EPS,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    ref = _sparse_reference(
        q,
        k,
        idx_q,
        idx_k,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    for got_tensor, ref_tensor in zip(got, ref):
        torch.testing.assert_close(got_tensor, ref_tensor, atol=3e-2, rtol=3e-2)

    torch.testing.assert_close(
        k_cache.index_select(0, out_cache_loc),
        ref[1].view(num_tokens, k_heads, head_dim),
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        v_cache.index_select(0, out_cache_loc),
        v.view(num_tokens, k_heads, head_dim),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        idx_k_cache.index_select(0, out_cache_loc),
        ref[3].view(num_tokens, 1, head_dim),
        atol=3e-2,
        rtol=3e-2,
    )


FP8_KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _make_sparse_pool(main_dtype, index_dtype, slots, k_heads, head_dim):
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    pool = MiniMaxSparseKVPool(
        size=slots,
        page_size=1,
        dtype=main_dtype,
        index_dtype=index_dtype,
        head_num=k_heads,
        head_dim=head_dim,
        idx_head_dim=head_dim,
        dense_layer_ids=[],
        sparse_layer_ids=[0],
        disable_value_sparse_layer_ids=[0],
        device=DEVICE,
        start_layer=0,
        end_layer=1,
    )
    # The unfused reference must take the pool's separate stores.
    pool.use_minimax_fused_kv_index_store = False
    return pool


@pytest.mark.parametrize(
    "main_dtype,index_dtype",
    [(d, torch.float8_e4m3fn) for d in FP8_KV_DTYPES]
    + [(d, torch.bfloat16) for d in FP8_KV_DTYPES]
    + [(torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("scales", [(None, None), (1.0, 1.0), (0.37, 1.7)])
@pytest.mark.parametrize("num_tokens", [1, 77])
@torch.inference_mode()
def test_sparse_cache_fusion_matches_unfused_pool_store(
    main_dtype, index_dtype, scales, num_tokens
):
    """With fp8 caches the fused store must write what norm+RoPE then the pool store write."""
    torch.manual_seed(num_tokens)
    dtype, q_heads, k_heads, idx_q_heads = torch.bfloat16, 16, 1, 4
    head_dim, rotary_dim, slots = 128, 64, 256
    q = torch.randn(num_tokens, q_heads * head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(num_tokens, k_heads * head_dim, device=DEVICE, dtype=dtype) * 8
    idx_q = torch.randn(num_tokens, idx_q_heads * head_dim, device=DEVICE, dtype=dtype)
    idx_k = torch.randn(num_tokens, head_dim, device=DEVICE, dtype=dtype)
    weights = [torch.randn(head_dim, device=DEVICE) for _ in range(4)]
    positions = torch.randint(0, 512, (num_tokens,), device=DEVICE)
    cos_sin_cache = torch.randn(512, rotary_dim, device=DEVICE, dtype=dtype)
    # Scattered slots; the last token is cuda-graph padding on slot 0.
    loc = torch.randperm(slots, device=DEVICE)[:num_tokens] + 1
    loc[-1] = 0
    norm_args = (*weights, positions, cos_sin_cache, EPS, head_dim, rotary_dim, True)

    fused = _make_sparse_pool(main_dtype, index_dtype, slots, k_heads, head_dim)
    k_cache, v_cache = fused.get_kv_buffer(0)
    idx_k_cache = fused.get_index_k_buffer(0)
    got = sparse_qk_index_gemma_rmsnorm_rope_cache(
        q,
        k,
        v,
        idx_q,
        idx_k,
        k_cache,
        v_cache,
        idx_k_cache,
        loc,
        *norm_args,
        k_scale=scales[0],
        v_scale=scales[1],
    )

    unfused = _make_sparse_pool(main_dtype, index_dtype, slots, k_heads, head_dim)
    ref = sparse_qk_index_gemma_rmsnorm_rope(q, k, idx_q, idx_k, *norm_args)
    unfused.set_fused_kv_index_buffer(
        SimpleNamespace(layer_id=0),
        loc,
        ref[1].view(num_tokens, k_heads, head_dim).clone(),
        v.view(num_tokens, k_heads, head_dim).clone(),
        ref[3].view(num_tokens, 1, head_dim),
        None,
        *scales,
    )
    torch.cuda.synchronize()

    for got_tensor, ref_tensor in zip(got, ref):
        assert torch.equal(got_tensor, ref_tensor)
    ref_k_cache, ref_v_cache = unfused.get_kv_buffer(0)
    assert torch.equal(k_cache.view(torch.uint8), ref_k_cache.view(torch.uint8))
    assert torch.equal(v_cache.view(torch.uint8), ref_v_cache.view(torch.uint8))
    got_idx = idx_k_cache.float()
    ref_idx = unfused.get_index_k_buffer(0).float()
    if index_dtype.itemsize == 1:
        # The fused index-K store casts fp32 straight to fp8 instead of rounding
        # through bf16 first, so a value can land one fp8 step (3 mantissa bits) away.
        one_step = torch.maximum(got_idx.abs(), ref_idx.abs()) / 8
        one_step += torch.finfo(index_dtype).tiny / 8
        assert ((got_idx - ref_idx).abs() <= one_step).all()
    else:
        assert torch.equal(got_idx, ref_idx)


@torch.inference_mode()
def test_sparse_cache_kernel_matches_the_per_head_kernel_on_a_prefill_chunk():
    """The fused kernel's bf16 outputs equal the per-head kernel's bit for bit, so its
    sum of squares keeps that add order; any other flips roundings at this size."""
    torch.manual_seed(5)
    num_tokens, q_heads, head_dim, rotary_dim = 4096, 16, 128, 64
    # Strided views of one fused projection row per token, as the M3.1 layer splits it.
    widths = [q_heads * head_dim] + [head_dim] * 4
    packed = torch.randn(num_tokens, sum(widths), device=DEVICE, dtype=torch.bfloat16)
    q, k, v, idx_q, idx_k = packed.split(widths, dim=-1)
    weights = [
        (torch.randn(head_dim, device=DEVICE) * 0.3).to(torch.bfloat16)
        for _ in range(4)
    ]
    positions = torch.randint(0, 4096, (num_tokens,), device=DEVICE)
    cos_sin_cache = torch.randn(4096, rotary_dim, device=DEVICE)
    norm_args = (*weights, positions, cos_sin_cache, EPS, head_dim, rotary_dim, True)
    caches = [
        torch.zeros(
            num_tokens + 1, 1, head_dim, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        for _ in range(3)
    ]
    out_cache_loc = torch.arange(1, num_tokens + 1, device=DEVICE)

    got = sparse_qk_index_gemma_rmsnorm_rope_cache(
        q, k, v, idx_q, idx_k, *caches, out_cache_loc, *norm_args
    )
    ref = sparse_qk_index_gemma_rmsnorm_rope(q, k, idx_q, idx_k, *norm_args)
    for got_tensor, ref_tensor in zip(got, ref):
        assert torch.equal(got_tensor, ref_tensor)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
