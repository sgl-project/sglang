"""Bit-exactness of the permuted -> expanded MoE activation gather.

The oracle is the torch reference FlashInfer uses for the same step
(``flashinfer/fused_moe/moe_lora_delta.py:271-274``): scatter the rows the
permutation map points at, leave every inactive slot (``perm < 0``) at zero. A
gather does no arithmetic, so every comparison here is ``torch.equal``.

Two properties beyond the reference are load-bearing for the caller and get
their own assertions:

- the destination arrives uninitialized, so it is NaN-poisoned before each call;
  a row the kernel forgets to write shows up as NaN rather than as luck.
- the permuted source is padded to ``max_num_padded_tokens_gemm1``, and the pad
  rows hold whatever the MoE workspace last left there. They are NaN-poisoned
  too, so an out-of-range read shows up instead of silently matching.
- that same expert-sorted tile padding puts live indices well past
  ``num_tokens * top_k``, so one case draws from a much larger source and poisons
  every row the map does not name -- a clamp or wrap on the source row is
  invisible to any fixture whose indices all fit in the destination.
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_gather_permuted import gather_permuted_activation
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

# I = 384 is Inkling at TP=8; 1, 17 and 1500 are non-power-of-two widths that
# straddle the BLOCK_I tiling (one partial tile, and more than one tile).
SHAPES = list(
    itertools.product(
        [1, 4, 63, 256],  # num_tokens
        [1, 2, 6],  # top_k
        [1, 17, 384, 1500],  # intermediate
    )
)


def _reference(
    activation_permuted: torch.Tensor,
    perm: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    idx = perm.to(torch.int64)
    valid = idx >= 0
    rows, inter = num_tokens * top_k, activation_permuted.shape[1]
    out = torch.zeros(
        rows, inter, dtype=activation_permuted.dtype, device=activation_permuted.device
    )
    out[valid] = activation_permuted[idx[valid]]
    return out.view(num_tokens, top_k, inter)


def _make_perm(
    rows: int, num_src_rows: int, inactive_frac: float, seed: int
) -> torch.Tensor:
    """A permutation-like map: distinct source rows, ``-1`` for inactive slots."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_src_rows, generator=g)[:rows].to(torch.int32)
    # At the decode shapes (rows = 1 or 2) a plain int(rows * frac) rounds to zero,
    # which would make every -1 assertion below compare two empty tensors.
    num_inactive = 0 if inactive_frac == 0 else max(1, int(rows * inactive_frac))
    if num_inactive:
        holes = torch.randperm(rows, generator=g)[:num_inactive]
        perm[holes] = -1
    return perm.cuda()


def _poisoned_source(
    num_src_rows: int, num_used_rows: int, inter: int, dtype: torch.dtype
) -> torch.Tensor:
    """Real rows up to ``num_used_rows``; NaN in the padding beyond it."""
    src = torch.randn(num_src_rows, inter, dtype=torch.float32, device="cuda").to(dtype)
    src[num_used_rows:] = float("nan")
    return src


@requires_cuda
@pytest.mark.parametrize("num_tokens,top_k,inter", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_matches_reference(num_tokens, top_k, inter, dtype):
    rows = num_tokens * top_k
    # The permuted buffer is padded well past the rows the map addresses.
    num_src_rows = rows + 128
    perm = _make_perm(rows, rows, inactive_frac=0.3, seed=num_tokens * 31 + top_k)
    assert (perm < 0).any(), "fixture never exercises the -1 branch"
    src = _poisoned_source(num_src_rows, rows, inter, dtype)

    out = torch.full(
        (num_tokens, top_k, inter), float("nan"), dtype=dtype, device="cuda"
    )
    got = gather_permuted_activation(src, perm, num_tokens, top_k, out=out)
    assert got.data_ptr() == out.data_ptr()

    ref = _reference(src, perm, num_tokens, top_k)
    assert torch.equal(got, ref)
    assert got.isfinite().all(), "a padded source row leaked into the destination"

    # The zero-fill contract, asserted on the bits and not through the reference.
    inactive = (perm < 0).view(num_tokens, top_k)
    assert torch.equal(got[inactive], torch.zeros_like(got[inactive])), (
        "inactive slots must gather as exact zeros"
    )


@requires_cuda
@pytest.mark.parametrize("inactive_frac", [0.0, 1.0])
def test_all_and_no_inactive(inactive_frac):
    num_tokens, top_k, inter = 33, 6, 384
    rows = num_tokens * top_k
    perm = _make_perm(rows, rows, inactive_frac=inactive_frac, seed=7)
    src = _poisoned_source(rows + 64, rows, inter, torch.bfloat16)

    out = torch.full(
        (num_tokens, top_k, inter), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    got = gather_permuted_activation(src, perm, num_tokens, top_k, out=out)

    assert torch.equal(got, _reference(src, perm, num_tokens, top_k))
    if inactive_frac == 1.0:
        assert torch.equal(got, torch.zeros_like(got))
    else:
        assert got.isfinite().all()


@requires_cuda
@pytest.mark.parametrize("num_tokens,top_k,inter", [(63, 6, 384), (4, 2, 17)])
def test_gathers_from_rows_beyond_the_destination(num_tokens, top_k, inter):
    """A live index is routinely larger than ``num_tokens * top_k``.

    The permuted buffer is expert-sorted and tile-padded (``max_num_padded_tokens_gemm1``
    strides each expert up to the tile), so the map names source rows well past the
    destination row count. Every fixture above draws its indices from ``[0, rows)``,
    which makes a clamp or a wrap on the source row invisible; here the poison sits
    on every row the map does NOT name, so one shows up as NaN.
    """
    rows = num_tokens * top_k
    num_src_rows = 4 * rows + 37
    g = torch.Generator(device="cpu").manual_seed(rows)
    perm = torch.randperm(num_src_rows, generator=g)[:rows].to(torch.int32)
    perm[torch.randperm(rows, generator=g)[: max(1, rows // 4)]] = -1
    assert (perm >= rows).any(), "fixture must name at least one high row"

    src = torch.randn(num_src_rows, inter, dtype=torch.float32, device="cuda").to(
        torch.bfloat16
    )
    unused = torch.ones(num_src_rows, dtype=torch.bool)
    unused[perm[perm >= 0].to(torch.int64)] = False
    src[unused.cuda()] = float("nan")

    perm = perm.cuda()
    out = torch.full(
        (num_tokens, top_k, inter), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    got = gather_permuted_activation(src, perm, num_tokens, top_k, out=out)
    assert got.isfinite().all(), "the gather read a row the map never named"
    assert torch.equal(got, _reference(src, perm, num_tokens, top_k))


@requires_cuda
def test_allocates_when_out_is_none():
    num_tokens, top_k, inter = 8, 4, 384
    rows = num_tokens * top_k
    perm = _make_perm(rows, rows, inactive_frac=0.5, seed=11)
    src = _poisoned_source(rows + 32, rows, inter, torch.bfloat16)

    got = gather_permuted_activation(src, perm, num_tokens, top_k)
    assert got.shape == (num_tokens, top_k, inter)
    assert got.dtype == src.dtype
    assert torch.equal(got, _reference(src, perm, num_tokens, top_k))


@requires_cuda
def test_rejects_fused_shared_expert_map():
    num_tokens, top_k, inter = 4, 6, 384
    src = torch.randn(64, inter, dtype=torch.bfloat16, device="cuda")
    # num_fused_shared_experts == 1 -> the map carries num_tokens * (top_k + 1).
    perm = torch.zeros(num_tokens * (top_k + 1), dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="num_fused_shared_experts"):
        gather_permuted_activation(src, perm, num_tokens, top_k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
