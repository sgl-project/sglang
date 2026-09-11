"""Skip absorb-q concat when nope/rope already abut in one fused tensor."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEADS = 64


def _split_fused(tokens: int, d_nope: int, d_rope: int):
    fused = torch.randn(tokens, HEADS, d_nope + d_rope, device=DEVICE, dtype=DTYPE)
    return fused, fused[..., :d_nope], fused[..., d_nope:]


@pytest.mark.parametrize("d_nope,d_rope", [(512, 64), (128, 64)])
def test_abutting_halves_alias_the_parent(d_nope: int, d_rope: int) -> None:
    fused, q_nope, q_rope = _split_fused(8, d_nope, d_rope)
    copied = torch.cat([q_nope, q_rope], dim=-1)
    aliased = concat_mla_absorb_q_general(q_nope, q_rope)

    assert torch.equal(aliased, copied)
    assert aliased.shape == copied.shape
    assert aliased.stride() == copied.stride()
    assert aliased.dtype == copied.dtype
    assert aliased.is_contiguous()
    assert aliased.data_ptr() == fused.data_ptr()
    assert aliased.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()


def test_separate_allocations_still_concat() -> None:
    q_nope = torch.randn(8, HEADS, 512, device=DEVICE, dtype=DTYPE)
    q_pe = torch.randn(8, HEADS, 64, device=DEVICE, dtype=DTYPE)
    result = concat_mla_absorb_q_general(q_nope, q_pe)
    assert torch.equal(result, torch.cat([q_nope, q_pe], dim=-1))
    assert result.data_ptr() != q_nope.data_ptr()


def test_non_adjacent_halves_still_concat() -> None:
    fused, nope, rope = _split_fused(8, 512, 64)
    reversed_cat = concat_mla_absorb_q_general(rope, nope)
    assert torch.equal(reversed_cat, torch.cat([rope, nope], dim=-1))
    assert reversed_cat.data_ptr() != fused.data_ptr()

    gapped = torch.randn(8, HEADS, 512 + 64 + 8, device=DEVICE, dtype=DTYPE)
    gapped_nope = gapped[..., :512]
    gapped_rope = gapped[..., 512 + 8 :]
    gapped_cat = concat_mla_absorb_q_general(gapped_nope, gapped_rope)
    assert torch.equal(gapped_cat, torch.cat([gapped_nope, gapped_rope], dim=-1))

    rope_from_start = fused[..., :64]
    start_cat = concat_mla_absorb_q_general(nope, rope_from_start)
    assert torch.equal(start_cat, torch.cat([nope, rope_from_start], dim=-1))
    assert start_cat.data_ptr() != fused.data_ptr()


def test_halves_from_different_parents_still_concat() -> None:
    fused_a, q_nope, _ = _split_fused(8, 512, 64)
    fused_b, _, q_rope = _split_fused(8, 512, 64)
    result = concat_mla_absorb_q_general(q_nope, q_rope)
    assert torch.equal(result, torch.cat([q_nope, q_rope], dim=-1))
    assert result.data_ptr() != fused_a.data_ptr()
    assert result.data_ptr() != fused_b.data_ptr()


def test_sparse_mla_output_is_bit_identical_with_the_view() -> None:
    pytest.importorskip("sgl_kernel")
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    tokens, kv_len, topk = 64, 256, 64
    _fused, q_nope, q_rope = _split_fused(tokens, 512, 64)
    kv = torch.randn(kv_len, 1, 576, device=DEVICE, dtype=DTYPE)
    indices = torch.randint(
        0, kv_len, (tokens, 1, topk), dtype=torch.int32, device=DEVICE
    )

    copied = torch.cat([q_nope, q_rope], dim=-1)
    aliased = concat_mla_absorb_q_general(q_nope, q_rope)
    assert copied.data_ptr() != aliased.data_ptr()

    out_copied, _, _ = flash_mla_sparse_fwd(
        q=copied, kv=kv, indices=indices, sm_scale=0.1, d_v=512
    )
    out_alias, _, _ = flash_mla_sparse_fwd(
        q=aliased, kv=kv, indices=indices, sm_scale=0.1, d_v=512
    )
    assert torch.equal(out_copied, out_alias)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
