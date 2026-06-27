"""End-to-end equivalence between USPAttention varlen path and SDPA reference.

Compares the production varlen path (``build_varlen_mask_meta`` +
``fused_pack_qkv`` + ``flash_attn_varlen_func`` + ``fused_scatter_to_padded``)
against ``torch.nn.functional.scaled_dot_product_attention`` with a broadcast
key mask, for inputs the gating in ``USPAttention.forward`` would accept.

Verifies the documented contract:
  * Valid (non-masked) query rows match SDPA within FA-vs-SDPA tolerance.
  * Masked query rows are exactly zero in the varlen path (differs from
    SDPA, which produces deterministic attention output at those rows).
"""

import unittest

import pytest
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    from sglang.jit_kernel.diffusion.triton.varlen_pack_pad import (
        fused_pack_qkv,
        fused_scatter_to_padded,
    )
    from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
    from sglang.jit_kernel.utils import get_ci_test_range
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        flash_attn as _fa_backend,
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import (
        build_varlen_mask_meta,
    )
else:
    fused_pack_qkv = None
    fused_scatter_to_padded = None
    flash_attn_varlen_func = None
    get_ci_test_range = lambda x, y=None: x
    _fa_backend = None
    build_varlen_mask_meta = None
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = get_ci_test_range([torch.bfloat16, torch.float16], [torch.bfloat16])
# (name, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens)
SHAPES = get_ci_test_range(
    [
        ("small_c2", 2, 64, 128, 4, 64, [32, 48]),
        ("prod_c2", 2, 256, 1024, 24, 128, [128, 200]),
        ("all_valid_b1", 1, 64, 128, 4, 64, [64]),
        ("zero_txt_one_batch", 2, 64, 128, 4, 64, [0, 32]),
    ],
    [
        ("small_c2", 2, 64, 128, 4, 64, [32, 48]),
    ],
)


def _build_mask(bs, s_txt, s_img, valid_txt_lens):
    s = s_txt + s_img
    mask = torch.zeros(bs, s, dtype=torch.bool, device=DEVICE)
    for b, vt in enumerate(valid_txt_lens):
        mask[b, :vt] = True
        mask[b, s_txt:] = True
    return mask


def _sdpa_with_key_mask(q, k, v, key_mask, softmax_scale):
    """Reference: SDPA with a ``[B, S]`` key mask broadcast to ``[B, 1, 1, S]``."""
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    mask = key_mask.to(dtype=q.dtype)[:, None, None, :]
    mask = (mask - 1.0) * torch.finfo(q.dtype).max
    out = F.scaled_dot_product_attention(
        q_,
        k_,
        v_,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


def _varlen_path(q, k, v, key_mask, softmax_scale):
    """Production varlen path matching USPAttention.forward masked branch."""
    bs, seq = q.shape[0], q.shape[1]
    meta = build_varlen_mask_meta(key_mask)
    indices = meta["indices"]
    if indices.shape[0] == 0:
        return torch.zeros_like(q)
    q_unpad, k_unpad, v_unpad = fused_pack_qkv(q, k, v, indices)
    out_unpad = flash_attn_varlen_func(
        q=q_unpad,
        k=k_unpad,
        v=v_unpad,
        cu_seqlens_q=meta["cu_seqlens"],
        cu_seqlens_k=meta["cu_seqlens"],
        max_seqlen_q=meta["max_seqlen"],
        max_seqlen_k=meta["max_seqlen"],
        softmax_scale=softmax_scale,
        causal=False,
        ver=_fa_backend.fa_ver,
    )
    return fused_scatter_to_padded(out_unpad, meta["inv_indices"], bs, seq)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", SHAPES, ids=lambda s: s[0] if isinstance(s, tuple) else str(s)
)
def test_varlen_path_matches_sdpa_on_valid_rows(dtype, shape):
    """Valid rows: varlen output ≈ SDPA output within FA tolerance."""
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(0)
    s = s_txt + s_img
    softmax_scale = head_dim**-0.5
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    q = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)

    out_sdpa = _sdpa_with_key_mask(q, k, v, mask, softmax_scale)
    out_varlen = _varlen_path(q, k, v, mask, softmax_scale)

    valid = mask[..., None, None].expand_as(out_sdpa)
    rtol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        out_sdpa[valid],
        out_varlen[valid],
        rtol=rtol,
        atol=atol,
    )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", SHAPES, ids=lambda s: s[0] if isinstance(s, tuple) else str(s)
)
def test_varlen_path_zeros_masked_rows(dtype, shape):
    """Masked rows: varlen path produces exact zeros (documented contract)."""
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(1)
    s = s_txt + s_img
    softmax_scale = head_dim**-0.5
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    q = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)

    out_varlen = _varlen_path(q, k, v, mask, softmax_scale)

    invalid = ~mask
    if invalid.any():
        assert (out_varlen[invalid] == 0).all(), "masked rows must be zero-filled"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
