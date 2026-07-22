"""Numerical correctness for fused varlen pack/scatter Triton kernels.

Bit-exact comparison against the equivalent PyTorch ops (index_select,
zeros + index_copy_) across bf16/fp16 and several shape/mask cases.
"""

import pytest
import torch

from sglang.jit_kernel.diffusion.triton.varlen_pack_pad import (
    build_inv_indices,
    fused_pack_qkv,
    fused_scatter_to_padded,
)
from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)
register_amd_ci(est_time=15, suite="nightly-amd-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = get_ci_test_range([torch.bfloat16, torch.float16], [torch.bfloat16])
# (bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens) tuples
SHAPES = get_ci_test_range(
    [
        # name, bs, s_txt, s_img, H, D, valid_txt_lens
        ("small_c2", 2, 64, 128, 4, 64, [32, 48]),
        ("prod_c2", 2, 256, 1024, 24, 128, [128, 200]),
        ("all_valid_b1", 1, 64, 128, 4, 64, [64]),
        ("all_valid_b4", 4, 64, 128, 4, 64, [64, 64, 64, 64]),
        ("c8_prod", 8, 256, 4096, 24, 128, [128, 200, 256, 100, 50, 256, 256, 50]),
        # one batch with zero valid text tokens (image side still valid)
        ("zero_txt_one_batch", 2, 64, 128, 4, 64, [0, 32]),
        # bs=1 with no text validity (only image rows packed)
        ("bs1_zero_txt", 1, 64, 128, 4, 64, [0]),
    ],
    [
        ("small_c2", 2, 64, 128, 4, 64, [32, 48]),
        ("prod_c2", 2, 256, 1024, 24, 128, [128, 200]),
        ("all_valid_b4", 4, 64, 128, 4, 64, [64, 64, 64, 64]),
    ],
)


def _build_mask(bs, s_txt, s_img, valid_txt_lens):
    s = s_txt + s_img
    mask = torch.zeros(bs, s, dtype=torch.bool, device=DEVICE)
    for b, vt in enumerate(valid_txt_lens):
        mask[b, :vt] = True
        mask[b, s_txt:] = True
    return mask


def _ref_pack(q, k, v, indices):
    bs, seq = q.shape[:2]
    flat = lambda t: t.reshape(bs * seq, *t.shape[2:])
    return (
        flat(q).index_select(0, indices),
        flat(k).index_select(0, indices),
        flat(v).index_select(0, indices),
    )


def _ref_scatter(out_unpad, indices, bs, seq):
    n_valid = indices.shape[0]
    _, num_heads, head_dim = out_unpad.shape
    flat = torch.zeros(
        bs * seq, num_heads, head_dim, dtype=out_unpad.dtype, device=DEVICE
    )
    flat.index_copy_(0, indices, out_unpad)
    return flat.view(bs, seq, num_heads, head_dim)


def _build_meta(mask):
    bs, seq = mask.shape
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    inv_indices = build_inv_indices(indices, bs * seq)
    return indices, inv_indices


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", SHAPES, ids=lambda s: s[0] if isinstance(s, tuple) else str(s)
)
def test_pack_matches_index_select(dtype, shape):
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(0)
    s = s_txt + s_img
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    indices, _ = _build_meta(mask)

    q = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)

    q_ref, k_ref, v_ref = _ref_pack(q, k, v, indices)
    q_fused, k_fused, v_fused = fused_pack_qkv(q, k, v, indices)

    # bit-exact: pack is pure gather, no math
    assert torch.equal(q_ref, q_fused)
    assert torch.equal(k_ref, k_fused)
    assert torch.equal(v_ref, v_fused)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", SHAPES, ids=lambda s: s[0] if isinstance(s, tuple) else str(s)
)
def test_scatter_matches_index_copy(dtype, shape):
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(1)
    s = s_txt + s_img
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    indices, inv_indices = _build_meta(mask)
    n_valid = indices.shape[0]

    out_unpad = torch.randn(n_valid, num_heads, head_dim, dtype=dtype, device=DEVICE)
    out_ref = _ref_scatter(out_unpad, indices, bs, s)
    out_fused = fused_scatter_to_padded(out_unpad, inv_indices, bs, s)

    # bit-exact: scatter is pure copy + zero-fill
    assert torch.equal(out_ref, out_fused)
    # Padding rows must be exactly zero
    invalid = ~mask
    if invalid.any():
        assert out_fused[invalid].abs().max().item() == 0.0


def test_pack_handles_non_contiguous_input():
    """Helper must accept non-contiguous Q/K/V (auto .contiguous() inside)."""
    torch.manual_seed(2)
    bs, s_txt, s_img, num_heads, head_dim = 2, 64, 128, 4, 64
    s = s_txt + s_img
    mask = _build_mask(bs, s_txt, s_img, [32, 48])
    indices, _ = _build_meta(mask)

    # Build non-contiguous tensors via permute
    qkv_pre = torch.randn(
        bs, num_heads, s, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    q = qkv_pre.permute(0, 2, 1, 3)
    k = torch.randn_like(qkv_pre).permute(0, 2, 1, 3)
    v = torch.randn_like(qkv_pre).permute(0, 2, 1, 3)
    assert not q.is_contiguous()

    q_ref, k_ref, v_ref = _ref_pack(
        q.contiguous(), k.contiguous(), v.contiguous(), indices
    )
    q_fused, k_fused, v_fused = fused_pack_qkv(q, k, v, indices)
    assert torch.equal(q_ref, q_fused)
    assert torch.equal(k_ref, k_fused)
    assert torch.equal(v_ref, v_fused)


def test_build_inv_indices_matches_manual():
    """build_inv_indices output should match the manual full+scatter form."""
    torch.manual_seed(3)
    bs, s = 2, 32
    mask = torch.bernoulli(torch.full((bs, s), 0.6, device=DEVICE)).to(torch.bool)
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    n_valid = indices.shape[0]

    manual = torch.full((bs * s,), -1, dtype=torch.int32, device=DEVICE)
    if n_valid > 0:
        manual[indices.long()] = torch.arange(n_valid, dtype=torch.int32, device=DEVICE)

    built = build_inv_indices(indices, bs * s)
    assert torch.equal(built, manual)


def test_empty_valid_set_handled():
    """All-False mask: pack returns empty tensors; scatter writes all zeros."""
    bs, s, num_heads, head_dim = 2, 16, 4, 64
    mask = torch.zeros(bs, s, dtype=torch.bool, device=DEVICE)
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    inv_indices = build_inv_indices(indices, bs * s)
    assert indices.numel() == 0

    q = torch.randn(bs, s, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    q_unpad, k_unpad, v_unpad = fused_pack_qkv(q, q.clone(), q.clone(), indices)
    assert q_unpad.shape == (0, num_heads, head_dim)
    assert k_unpad.shape == (0, num_heads, head_dim)
    assert v_unpad.shape == (0, num_heads, head_dim)

    out_padded = fused_scatter_to_padded(q_unpad, inv_indices, bs, s)
    assert out_padded.shape == (bs, s, num_heads, head_dim)
    assert out_padded.abs().max().item() == 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
