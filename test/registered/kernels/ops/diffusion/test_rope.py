"""``diffusion.rope``: rotary embeddings and the QK-norm chains fused into them.

Two families with different oracles:

- ``fused_inplace_qknorm_rope`` / ``fused_qknorm_rope_pack_kv`` are compared
  against the *split* baseline (a separate qknorm kernel plus FlashInfer or
  sgl_kernel RoPE).  In the default mode the two differ by about one bf16
  rounding step, so those cases use a tolerance; with
  ``round_norm_before_rope=True`` the fused kernel reproduces the split
  rounding exactly and ``torch.equal`` applies. Full-width interleaved caches
  use the Diffusers float32 RoPE chain as their oracle.
The LTX-2 split-RoPE kernel lives in ``test_rope_ltx2.py``: it is validated on
B200 and registered on that lane alone, which the cases here cannot share --
their oracle is the *split* baseline (a separate qknorm kernel plus sgl_kernel
or FlashInfer RoPE), whose dispatch differs on Blackwell, so the bit-exact
assertions below do not hold there.
"""

import itertools
import sys

import pytest
import torch
import triton

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
    fused_qknorm_rope_pack_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=18, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1, which
# expands the get_ci_test_range sweeps below.
register_cuda_ci(est_time=220, stage="nightly", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0
ATOL = 8e-2
RTOL = 1e-2


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def split_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    fused_inplace_qknorm(q, k, q_weight, k_weight)
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions.long(),
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=q.shape[-1],
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def fused_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=is_neox,
        rope_dim=cos_sin_cache.shape[-1],
    )


def test_qknorm_rope_rejects_unsupported_dtypes() -> None:
    assert not can_use_fused_inplace_qknorm_rope(128, 128, False, torch.float32)
    assert not can_use_fused_inplace_qknorm_rope(
        128, 128, False, torch.bfloat16, torch.float64
    )


_FULL_BS_LIST = [2**n for n in range(13)]
_FULL_BS_LIST += [x + 1 for x in _FULL_BS_LIST]
_FULL_HEADS_LIST = [8, 16, 24, 32]
_FULL_HEAD_DIM_LIST = [64, 128, 256]
IS_NEOX_LIST = [False, True]
POSITION_DTYPES = [torch.int32, torch.int64]
ROPE_DIM_CHOICES = {
    64: [64],
    128: [64, 128],
    256: [64, 128, 256],
}
QKNORM_ROPE_CASES = get_ci_test_range(
    list(
        itertools.product(
            _FULL_BS_LIST,
            _FULL_HEADS_LIST,
            _FULL_HEAD_DIM_LIST,
            IS_NEOX_LIST,
            POSITION_DTYPES,
        )
    ),
    [
        (1, 8, 64, False, torch.int32),
        (9, 24, 128, True, torch.int64),
        (129, 8, 256, True, torch.int32),
        (257, 24, 64, False, torch.int64),
        (2049, 8, 128, True, torch.int32),
        (4097, 24, 256, False, torch.int64),
        (1, 24, 64, True, torch.int64),
        (129, 8, 128, False, torch.int32),
        (2049, 24, 256, True, torch.int64),
        (4097, 8, 64, False, torch.int32),
    ],
)


@pytest.mark.parametrize(
    "batch_size,num_heads,head_dim,is_neox,position_dtype",
    QKNORM_ROPE_CASES,
)
def test_qknorm_rope(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    is_neox: bool,
    position_dtype: torch.dtype,
) -> None:
    rope_dims = ROPE_DIM_CHOICES[head_dim]
    for rope_dim in rope_dims:
        if is_neox:
            elems_per_thread = head_dim // 32
            rotary_lanes = rope_dim // elems_per_thread
            if rotary_lanes < 2 or rotary_lanes % 2:
                continue

        q = torch.randn(batch_size, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        k = torch.randn(batch_size, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
        k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
        positions = torch.randint(
            0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=position_dtype
        )
        cos_sin_cache = create_cos_sin_cache(rope_dim)

        q_ref, k_ref = q.clone(), k.clone()
        q_fused, k_fused = q.clone(), k.clone()

        split_qknorm_rope(
            q_ref, k_ref, q_weight, k_weight, cos_sin_cache, positions, is_neox
        )
        fused_qknorm_rope(
            q_fused, k_fused, q_weight, k_weight, cos_sin_cache, positions, is_neox
        )

        # The split baseline mixes a separate BF16 qknorm kernel with FlashInfer RoPE,
        # which differs from the fused path by about one BF16 rounding step on H200.
        triton.testing.assert_close(q_ref, q_fused, atol=ATOL, rtol=RTOL)
        triton.testing.assert_close(k_ref, k_fused, atol=ATOL, rtol=RTOL)


def test_qknorm_rope_preserves_split_bf16_rounding() -> None:
    from sgl_kernel import rotary_embedding

    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    num_tokens, num_heads, head_dim, rope_dim = 257, 28, 128, 96
    inner_dim = num_heads * head_dim
    qkv = torch.randn(
        num_tokens,
        3 * inner_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    cos_sin_cache = create_cos_sin_cache(rope_dim, num_tokens).to(DTYPE)

    qkv_ref, qkv_fused = qkv.clone(), qkv.clone()
    q_ref, k_ref, _ = qkv_ref.split(inner_dim, dim=-1)
    q_fused, k_fused, _ = qkv_fused.split(inner_dim, dim=-1)
    q_ref = q_ref.view(num_tokens, num_heads, head_dim)
    k_ref = k_ref.view(num_tokens, num_heads, head_dim)
    q_fused = q_fused.view(num_tokens, num_heads, head_dim)
    k_fused = k_fused.view(num_tokens, num_heads, head_dim)

    fused_inplace_qknorm(q_ref, k_ref, q_weight, k_weight, eps=1e-5)
    rotary_embedding(
        positions,
        q_ref.view(num_tokens, -1),
        k_ref.view(num_tokens, -1),
        head_dim,
        cos_sin_cache,
        True,
    )
    fused_inplace_qknorm_rope(
        q_fused,
        k_fused,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        eps=1e-5,
        rope_dim=rope_dim,
        round_norm_before_rope=True,
    )

    assert torch.equal(q_ref, q_fused)
    assert torch.equal(k_ref, k_fused)


def test_qknorm_rope_preserves_full_width_neox_cache() -> None:
    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    num_tokens, num_heads, head_dim = 257, 32, 128
    q = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    cos = torch.randn(num_tokens, head_dim, device=DEVICE, dtype=DTYPE)
    sin = torch.randn_like(cos)
    cache = torch.cat((cos, sin), dim=-1).contiguous()

    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm(q_ref, k_ref, q_weight, k_weight, eps=1e-6)
    half = head_dim // 2
    q1, q2 = q_ref[..., :half], q_ref[..., half:]
    k1, k2 = k_ref[..., :half], k_ref[..., half:]
    q_ref = torch.cat((-q2, q1), dim=-1) * sin[:, None, :] + q_ref * cos[:, None, :]
    k_ref = torch.cat((-k2, k1), dim=-1) * sin[:, None, :] + k_ref * cos[:, None, :]

    fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cache,
        positions,
        is_neox=True,
        eps=1e-6,
        round_norm_before_rope=True,
        cache_has_full_width=True,
    )

    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qknorm_rope_preserves_full_width_interleaved_cache(
    dtype: torch.dtype,
) -> None:
    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    num_tokens, num_heads, head_dim = 257, 24, 128
    q = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=dtype)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=dtype)
    positions = torch.randperm(num_tokens, device=DEVICE, dtype=torch.int64)
    cos = torch.randn(num_tokens, head_dim, device=DEVICE)
    sin = torch.randn_like(cos)
    cache = torch.cat((cos, sin), dim=-1).contiguous()

    def apply_interleaved_rope(x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack((-x_imag, x_real), dim=-1).flatten(-2)
        selected_cos = cos[positions, None]
        selected_sin = sin[positions, None]
        return (x.float() * selected_cos + x_rotated * selected_sin).to(dtype)

    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm(q_ref, k_ref, q_weight, k_weight, eps=1e-6)
    q_ref = apply_interleaved_rope(q_ref)
    k_ref = apply_interleaved_rope(k_ref)

    fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cache,
        positions,
        is_neox=False,
        eps=1e-6,
        round_norm_before_rope=True,
        cache_has_full_width=True,
    )

    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)


def test_qknorm_rope_requires_opt_in_for_strided_packed_gqa() -> None:
    from sglang.multimodal_gen.runtime.layers.layernorm import (
        RMSNorm,
        apply_qk_norm_rope,
    )

    num_tokens, num_q_heads, num_kv_heads, head_dim = 257, 32, 8, 128
    num_heads = num_q_heads + 2 * num_kv_heads
    qkv = torch.randn(1, num_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    cos_sin_cache = create_cos_sin_cache(head_dim, num_tokens)

    q_ref = qkv[:, :, :num_q_heads].contiguous()
    k_ref = qkv[:, :, num_q_heads : num_q_heads + num_kv_heads].contiguous()
    q_norm = RMSNorm(head_dim, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    q_norm.weight.data.copy_(q_weight)
    k_norm.weight.data.copy_(k_weight)
    qkv_default = qkv.clone()
    q_default = qkv_default[:, :, :num_q_heads]
    k_default = qkv_default[:, :, num_q_heads : num_q_heads + num_kv_heads]
    q_default_out, k_default_out = apply_qk_norm_rope(
        q=q_default,
        k=k_default,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        positions=positions,
    )
    assert q_default_out.data_ptr() != q_default.data_ptr()
    assert k_default_out.data_ptr() != k_default.data_ptr()

    qkv_fused = qkv.clone()
    q_fused = qkv_fused[:, :, :num_q_heads]
    k_fused = qkv_fused[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_before = qkv_fused[:, :, num_q_heads + num_kv_heads :].clone()

    fused_inplace_qknorm_rope(
        q_ref.view(-1, num_q_heads, head_dim),
        k_ref.view(-1, num_kv_heads, head_dim),
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        rope_dim=head_dim,
    )
    q_out, k_out = apply_qk_norm_rope(
        q=q_fused,
        k=k_fused,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        positions=positions,
        allow_strided_qk=True,
    )

    assert q_out.data_ptr() == q_fused.data_ptr()
    assert k_out.data_ptr() == k_fused.data_ptr()
    assert torch.equal(q_ref, q_out)
    assert torch.equal(k_ref, k_out)
    assert torch.equal(v_before, qkv_fused[:, :, num_q_heads + num_kv_heads :])


def test_qknorm_rope_pack_kv_matches_separate_ops() -> None:

    batch_size = 2
    prefix_tokens, suffix_tokens = 17, 257
    num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    num_heads = num_q_heads + 2 * num_kv_heads
    qkv = torch.randn(
        batch_size,
        suffix_tokens,
        num_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    prefix_qkv = torch.randn(
        batch_size,
        prefix_tokens,
        num_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    k_prefix = prefix_qkv[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_prefix = prefix_qkv[:, :, num_q_heads + num_kv_heads :]
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(
        batch_size * suffix_tokens, device=DEVICE, dtype=torch.int64
    )
    cos_sin_cache = create_cos_sin_cache(head_dim, batch_size * suffix_tokens)

    qkv_ref = qkv.clone()
    q_ref = qkv_ref[:, :, :num_q_heads]
    k_ref = qkv_ref[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_ref = qkv_ref[:, :, num_q_heads + num_kv_heads :]
    fused_inplace_qknorm_rope(
        q_ref.view(-1, num_q_heads, head_dim),
        k_ref.view(-1, num_kv_heads, head_dim),
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        rope_dim=head_dim,
    )
    packed_k_ref = torch.cat([k_prefix, k_ref], dim=1)
    packed_v_ref = torch.cat([v_prefix, v_ref], dim=1)

    qkv_fused = qkv.clone()
    q_fused = qkv_fused[:, :, :num_q_heads]
    k_fused = qkv_fused[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_fused = qkv_fused[:, :, num_q_heads + num_kv_heads :]
    packed_kv = torch.empty(
        2,
        batch_size,
        prefix_tokens + suffix_tokens,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    fused_qknorm_rope_pack_kv(
        q_fused,
        k_fused,
        v_fused,
        k_prefix,
        v_prefix,
        packed_kv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        rope_dim=head_dim,
    )

    assert torch.equal(q_ref, q_fused)
    assert torch.equal(packed_k_ref, packed_kv[0])
    assert torch.equal(packed_v_ref, packed_kv[1])


def test_qknorm_rope_pack_kv_preserves_split_bf16_rounding() -> None:
    from sgl_kernel import rotary_embedding

    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    batch_size = 1
    prefix_tokens, suffix_tokens = 17, 1024
    num_q_heads, num_kv_heads, head_dim = 32, 8, 64
    num_heads = num_q_heads + 2 * num_kv_heads
    qkv = torch.randn(
        batch_size,
        suffix_tokens,
        num_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    prefix_qkv = torch.randn(
        batch_size,
        prefix_tokens,
        num_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    k_prefix = prefix_qkv[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_prefix = prefix_qkv[:, :, num_q_heads + num_kv_heads :]
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(
        batch_size * suffix_tokens, device=DEVICE, dtype=torch.int64
    )
    cos_sin_cache = create_cos_sin_cache(head_dim, batch_size * suffix_tokens).to(DTYPE)

    qkv_ref = qkv.clone()
    q_ref = qkv_ref[:, :, :num_q_heads]
    k_ref = qkv_ref[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_ref = qkv_ref[:, :, num_q_heads + num_kv_heads :]
    fused_inplace_qknorm(
        q_ref.view(-1, num_q_heads, head_dim),
        k_ref.view(-1, num_kv_heads, head_dim),
        q_weight,
        k_weight,
        eps=1e-6,
    )
    rotary_embedding(
        positions,
        q_ref.view(-1, num_q_heads * head_dim),
        k_ref.view(-1, num_kv_heads * head_dim),
        head_dim,
        cos_sin_cache,
        True,
    )
    packed_k_ref = torch.cat([k_prefix, k_ref], dim=1)
    packed_v_ref = torch.cat([v_prefix, v_ref], dim=1)

    qkv_fused = qkv.clone()
    q_fused = qkv_fused[:, :, :num_q_heads]
    k_fused = qkv_fused[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v_fused = qkv_fused[:, :, num_q_heads + num_kv_heads :]
    packed_kv = torch.empty(
        2,
        batch_size,
        prefix_tokens + suffix_tokens,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )
    fused_qknorm_rope_pack_kv(
        q_fused,
        k_fused,
        v_fused,
        k_prefix,
        v_prefix,
        packed_kv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        rope_dim=head_dim,
        round_norm_before_rope=True,
    )

    assert torch.equal(q_ref, q_fused)
    assert torch.equal(packed_k_ref, packed_kv[0])
    assert torch.equal(packed_v_ref, packed_kv[1])


def test_qknorm_rope_accepts_empty_token_dimension() -> None:

    num_heads, head_dim = 8, 128
    q = torch.empty(0, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.empty_like(q)
    weight = torch.ones(head_dim, device=DEVICE, dtype=DTYPE)
    cache = create_cos_sin_cache(head_dim, 1)
    positions = torch.empty(0, device=DEVICE, dtype=torch.int64)

    fused_inplace_qknorm_rope(
        q,
        k,
        weight,
        weight,
        cache,
        positions,
        is_neox=False,
        rope_dim=head_dim,
    )
    assert q.numel() == k.numel() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
