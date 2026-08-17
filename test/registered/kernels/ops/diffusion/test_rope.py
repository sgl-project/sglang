"""``diffusion.rope``: rotary embeddings and the QK-norm chains fused into them.

Two families with different oracles:

- ``fused_inplace_qknorm_rope`` / ``fused_qknorm_rope_pack_kv`` are compared
  against the *split* baseline (a separate qknorm kernel plus FlashInfer or
  sgl_kernel RoPE).  In the default mode the two differ by about one bf16
  rounding step, so those cases use a tolerance; with
  ``round_norm_before_rope=True`` the fused kernel reproduces the split
  rounding exactly and ``torch.equal`` applies.
- The LTX-2 split-RoPE CUDA kernel is validated on B200 only (guarded per
  test, not per module, so the rest of this file still runs elsewhere).
"""

import itertools
import sys

import pytest
import torch
import torch.nn.functional as F
import triton

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    can_use_ltx2_qknorm_split_rope_cuda,
    fused_inplace_qknorm_rope,
    fused_qknorm_rope_pack_kv,
    ltx2_qknorm_split_rope_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=52, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
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
BF16_FUSED_ATOL = 1.6e-1


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


BS_LIST = [2**n for n in range(13)]
BS_LIST += [x + 1 for x in BS_LIST]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 129, 257, 2049, 4097])
HEADS_LIST = get_ci_test_range([8, 16, 24, 32], [8, 24])
HEAD_DIM_LIST = get_ci_test_range([64, 128, 256], [64, 128, 256])
IS_NEOX_LIST = [False, True]
POSITION_DTYPES = [torch.int32, torch.int64]
ROPE_DIM_CHOICES = {
    64: [64],
    128: [64, 128],
    256: [64, 128, 256],
}


@pytest.mark.parametrize(
    "batch_size,num_heads,head_dim,is_neox,position_dtype",
    list(
        itertools.product(
            BS_LIST,
            HEADS_LIST,
            HEAD_DIM_LIST,
            IS_NEOX_LIST,
            POSITION_DTYPES,
        )
    ),
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


# -------------------------------------------------------------------------
# LTX-2 split RoPE (B200)
# -------------------------------------------------------------------------


def _require_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("LTX2 QKNorm split-RoPE CUDA path is validated on B200")


def _ltx2_make_cos_sin(
    batch: int, seq_len: int, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = head_dim // 2
    cos = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    sin = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    return cos, sin


def _apply_split_rotary_ref(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_dtype = x.dtype
    batch = x.shape[0]
    _, num_heads, seq_len, _ = cos.shape
    x = x.reshape(batch, seq_len, num_heads, -1).swapaxes(1, 2)
    last = x.shape[-1]
    half = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, half)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, second_x)
    out[..., 1:, :].addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last)
    return out.swapaxes(1, 2).reshape(batch, seq_len, -1).to(dtype=x_dtype)


def _ltx2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rms_norm isn't autocast fp32-preserving, so feed fp32 inputs directly
    # to keep the normalized value unrounded until the final RoPE output.
    q_norm = F.rms_norm(q.float(), (q.shape[-1],), q_weight.float(), eps)
    k_norm = F.rms_norm(k.float(), (k.shape[-1],), k_weight.float(), eps)
    q_ref = _apply_split_rotary_ref(q_norm, q_cos, q_sin)
    k_ref = _apply_split_rotary_ref(k_norm, k_cos, k_sin)
    return q_ref.to(dtype=torch.bfloat16), k_ref.to(dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "batch,q_seq,k_seq,num_heads,head_dim",
    [
        (1, 3, 3, 32, 128),
        (1, 5, 2, 32, 64),
        (2, 4, 3, 32, 64),
    ],
)
def test_ltx2_qknorm_split_rope_matches_torch_exactly(
    batch: int, q_seq: int, k_seq: int, num_heads: int, head_dim: int
) -> None:
    _require_b200()
    torch.cuda.manual_seed(20260630)
    hidden = num_heads * head_dim
    eps = 1e-6
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    q_ref, k_ref = _ltx2_reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, eps
    )
    q_out, k_out = ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        eps=eps,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


def test_ltx2_qknorm_split_rope_rejects_unsupported_inputs() -> None:
    _require_b200()
    torch.cuda.manual_seed(20260630)
    q = torch.randn((1, 3, 4096), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_cos, q_sin = _ltx2_make_cos_sin(1, 3, 32, 128)
    q_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
    )
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q.float(),
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
    )
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos.transpose(-1, -2),
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
    )


def test_ltx2_qknorm_split_rope_custom_op_torch_compile_fullgraph() -> None:
    _require_b200()
    torch.cuda.manual_seed(20260630)
    batch, q_seq, k_seq, num_heads, head_dim = 1, 3, 2, 32, 64
    hidden = num_heads * head_dim
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    def fn(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight):
        return ltx2_qknorm_split_rope_cuda(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            eps=1e-6,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    compiled = torch.compile(fn, fullgraph=True)
    q_out, k_out = compiled(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight)
    q_ref, k_ref = _ltx2_reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, 1e-6
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
