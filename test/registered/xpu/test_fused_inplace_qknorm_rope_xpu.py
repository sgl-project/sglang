"""``sgl_kernel.fused_inplace_qknorm_rope`` on XPU: fused per-head RMSNorm + RoPE from a cos/sin cache.

The oracle is the split path (bf16 RMSNorm, then fp32-cache RoPE); the fused
kernel skips the intermediate bf16 rounding, so the two differ by about one
bf16 rounding step and are compared with a tolerance.
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

if is_xpu():
    from sgl_kernel import (
        fused_inplace_qknorm_rope,
        fused_qk_rope_with_cos_sin_cache_inplace,
    )

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.models.utils import apply_qk_norm

pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU required")

DEVICE = "xpu"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0
EPS = 1e-6
ATOL = 8e-2
RTOL = 1e-2
# XPU kernels run one 16-lane sub-group per head.
SUB_GROUP_SIZE = 16


def create_cos_sin_cache(rotary_dim: int, max_position: int = MAX_SEQ_LEN):
    inv_freq = 1.0 / (
        ROPE_BASE
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def split_qknorm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, is_neox):
    head_dim = q.shape[-1]
    for x, weight in ((q, q_weight), (k, k_weight)):
        norm = RMSNorm(head_dim, eps=EPS).to(DEVICE).to(DTYPE)
        norm.weight.data.copy_(weight)
        x.copy_(norm(x.reshape(-1, head_dim)).view(x.shape))
    rope_dim = cos_sin_cache.shape[-1]
    fused_qk_rope_with_cos_sin_cache_inplace(
        q[..., :rope_dim],
        k[..., :rope_dim],
        cos_sin_cache,
        positions,
        rope_dim,
        is_neox,
    )


def fused_qknorm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, is_neox):
    fused_inplace_qknorm_rope(
        q=q,
        k=k,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        is_neox=is_neox,
        eps=EPS,
        head_dim=q.shape[-1],
        rope_dim=cos_sin_cache.shape[-1],
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
    for rope_dim in ROPE_DIM_CHOICES[head_dim]:
        if is_neox:
            rotary_lanes = rope_dim // (head_dim // SUB_GROUP_SIZE)
            if rotary_lanes < 2 or rotary_lanes & (rotary_lanes - 1):
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

        torch.testing.assert_close(q_ref, q_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(k_ref, k_fused, atol=ATOL, rtol=RTOL)


# (num_q_heads, num_kv_heads): MHA, GQA, MQA.
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 8), (32, 4), (8, 1)])
@pytest.mark.parametrize(
    "head_dim,rope_dim", [(64, 64), (128, 64), (128, 128), (256, 128)]
)
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("num_tokens", [1, 7, 1024])
def test_qknorm_rope_on_packed_qkv_views(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    num_tokens: int,
) -> None:
    """q/k passed as strided views of a packed qkv are updated in place, and v is left intact."""
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, device=DEVICE, dtype=DTYPE)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, MAX_SEQ_LEN, (num_tokens,), device=DEVICE)
    cos_sin_cache = create_cos_sin_cache(rope_dim)

    q_ref = qkv[:, :q_size].reshape(num_tokens, num_q_heads, head_dim).clone()
    k_ref = (
        qkv[:, q_size : q_size + kv_size]
        .reshape(num_tokens, num_kv_heads, head_dim)
        .clone()
    )
    split_qknorm_rope(
        q_ref, k_ref, q_weight, k_weight, cos_sin_cache, positions, is_neox
    )

    fused = qkv.clone()
    q, k, v = fused.split([q_size, kv_size, kv_size], dim=-1)
    fused_qknorm_rope(
        q.view(-1, num_q_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox,
    )

    torch.testing.assert_close(q.view(q_ref.shape), q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k.view(k_ref.shape), k_ref, atol=ATOL, rtol=RTOL)
    assert torch.equal(v, qkv[:, q_size + kv_size :])


YARN_SCALING = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}


@pytest.mark.parametrize("rope_scaling", [None, YARN_SCALING], ids=["default", "yarn"])
@pytest.mark.parametrize("head_dim,rotary_dim", [(128, 128), (128, 64), (256, 256)])
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 4), (8, 1)])
@pytest.mark.parametrize("num_tokens", [1, 257])
def test_qknorm_rope_matches_model_unfused_path(
    rope_scaling,
    head_dim: int,
    rotary_dim: int,
    is_neox: bool,
    num_q_heads: int,
    num_kv_heads: int,
    num_tokens: int,
) -> None:
    """ Test for the fused call fed from a model's RotaryEmbedding matches its unfused apply_qk_norm + rotary_emb path."""
    max_position = 131072
    rotary_emb = get_rope(
        head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=1e6,
        is_neox_style=is_neox,
        rope_scaling=rope_scaling,
    ).to(DEVICE)
    q_norm = RMSNorm(head_dim, eps=EPS).to(DEVICE).to(DTYPE)
    k_norm = RMSNorm(head_dim, eps=EPS).to(DEVICE).to(DTYPE)
    q_norm.weight.data.normal_(mean=1.0, std=0.3)
    k_norm.weight.data.normal_(mean=1.0, std=0.3)

    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, max_position, (num_tokens,), device=DEVICE)

    q_ref, k_ref, _ = qkv.clone().split([q_size, kv_size, kv_size], dim=-1)
    q_ref, k_ref = apply_qk_norm(
        q=q_ref, k=k_ref, q_norm=q_norm, k_norm=k_norm, head_dim=head_dim
    )
    q_ref, k_ref = rotary_emb(positions, q_ref, k_ref)

    fused = qkv.clone()
    q, k, v = fused.split([q_size, kv_size, kv_size], dim=-1)
    fused_inplace_qknorm_rope(
        q=q.view(-1, num_q_heads, head_dim),
        k=k.view(-1, num_kv_heads, head_dim),
        q_weight=q_norm.weight,
        k_weight=k_norm.weight,
        cos_sin_cache=rotary_emb.cos_sin_cache,
        positions=positions,
        is_neox=rotary_emb.is_neox_style,
        eps=q_norm.variance_epsilon,
        head_dim=head_dim,
        rope_dim=rotary_emb.rotary_dim,
    )

    torch.testing.assert_close(q, q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k, k_ref, atol=ATOL, rtol=RTOL)
    assert torch.equal(v, qkv[:, q_size + kv_size :])


# rope_dim spanning 3 rotary lanes (head_dim // 16 elements per lane).
@pytest.mark.parametrize("head_dim,rope_dim", [(64, 12), (128, 24), (256, 48)])
def test_qknorm_rope_rejects_neox_non_pow2_rotary_lanes(
    head_dim: int, rope_dim: int
) -> None:
    """NeoX pairs lanes by XOR shuffle, so a non-power-of-2 lane count must raise, not mispair."""
    q = torch.randn(4, 8, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(4, 2, head_dim, device=DEVICE, dtype=DTYPE)
    weight = torch.ones(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(4, device=DEVICE)

    with pytest.raises(RuntimeError, match="power of 2"):
        fused_qknorm_rope(
            q, k, weight, weight, create_cos_sin_cache(rope_dim), positions, True
        )


def test_qknorm_rope_accepts_empty_token_dimension() -> None:
    num_heads, head_dim = 8, 128
    q = torch.empty(0, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.empty_like(q)
    weight = torch.ones(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.empty(0, device=DEVICE, dtype=torch.int64)

    fused_qknorm_rope(
        q, k, weight, weight, create_cos_sin_cache(head_dim, 1), positions, False
    )
    torch.xpu.synchronize()
    assert q.numel() == k.numel() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
