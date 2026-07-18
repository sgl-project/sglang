"""fused_extend_sconv_metadata must be bit-identical to the unfused prep.

The unfused reference is the exact op sequence _prepare_extend_common_metadata
+ precompute_helion_extend_metadata used to launch: zeros + cumsum + slice-copy
(or arange + ones for verify) + the has_initial_state compare, then != PAD, &,
clamp, long, to(int64), arange, searchsorted, clamp, to(int32).
"""

import pytest
import torch

from sglang.srt.models.inkling_common.kernels.sconv import (
    HIS_ONES,
    HIS_PREFIX,
    HIS_SEQ_MINUS_EXT,
    HIS_ZEROS,
    PAD_SLOT_ID,
    fused_extend_sconv_metadata,
    precompute_helion_extend_metadata,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

# cross si tiles (BLOCK_T=256) and the single-tile B bound
BATCH_SIZES = [1, 2, 7, 64, 257, 1023]


def _ref_extend(B, extend_seq_lens, his_mode, his_src, cache_indices, T):
    device = cache_indices.device
    query_start_loc = torch.zeros(B + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = extend_seq_lens.cumsum(dim=0)
    if his_mode == HIS_ZEROS:
        has_initial_state = torch.zeros(B, dtype=torch.bool, device=device)
    elif his_mode == HIS_PREFIX:
        has_initial_state = his_src > 0
    else:  # HIS_SEQ_MINUS_EXT
        has_initial_state = (his_src[:B] - extend_seq_lens) > 0
    meta = precompute_helion_extend_metadata(
        B=B,
        T=T,
        W=4,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        query_start_loc=query_start_loc,
    )
    return query_start_loc, has_initial_state, meta


def _ref_verify(B, draft_token_num, cache_indices):
    device = cache_indices.device
    query_start_loc = torch.arange(
        0, (B + 1) * draft_token_num, draft_token_num, dtype=torch.int32, device=device
    )
    has_initial_state = torch.ones(B, dtype=torch.bool, device=device)
    meta = precompute_helion_extend_metadata(
        B=B,
        T=B * draft_token_num,
        W=4,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        query_start_loc=query_start_loc,
    )
    return query_start_loc, has_initial_state, meta


def _assert_equal(got, ref):
    for tag, g, r in (
        ("query_start_loc", got[0], ref[0]),
        ("has_initial_state", got[1], ref[1]),
        ("cache_mask", got[2]["cache_mask"], ref[2]["cache_mask"]),
        ("safe_idx", got[2]["safe_idx"], ref[2]["safe_idx"]),
        ("cu", got[2]["cu"], ref[2]["cu"]),
        ("si", got[2]["si"], ref[2]["si"]),
    ):
        assert g.dtype == r.dtype, (tag, g.dtype, r.dtype)
        assert g.shape == r.shape, (tag, g.shape, r.shape)
        assert torch.equal(g, r), tag


def _cache_indices(b, idx_dtype):
    ci = torch.randint(0, 4096, (b,), dtype=idx_dtype, device="cuda")
    pad = torch.rand(b, device="cuda") < 0.25
    ci[pad] = PAD_SLOT_ID
    return ci


@requires_cuda
@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("his_mode", [HIS_ZEROS, HIS_PREFIX, HIS_SEQ_MINUS_EXT])
@pytest.mark.parametrize("lens_dtype", [torch.int32, torch.int64])
def test_extend_matches_unfused(b, his_mode, lens_dtype):
    torch.manual_seed(b * 10 + his_mode)
    lens = torch.randint(0, 33, (b,), dtype=lens_dtype, device="cuda")
    lens[torch.rand(b, device="cuda") < 0.2] = 0  # zero-length sequences
    T = int(lens.sum().item())
    cache_indices = _cache_indices(b, torch.int32)
    if his_mode == HIS_PREFIX:
        his_src = torch.randint(0, 3, (b,), dtype=lens_dtype, device="cuda")
    elif his_mode == HIS_SEQ_MINUS_EXT:
        his_src = lens + torch.randint(0, 2, (b,), dtype=lens_dtype, device="cuda")
    else:
        his_src = None

    ref = _ref_extend(b, lens, his_mode, his_src, cache_indices, T)
    got = fused_extend_sconv_metadata(
        B=b,
        T=T,
        cache_indices=cache_indices,
        his_mode=his_mode,
        extend_seq_lens=lens,
        his_src=his_src,
    )
    assert got is not None
    _assert_equal(got, ref)


@requires_cuda
@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("draft_token_num", [1, 9])
def test_verify_matches_unfused(b, draft_token_num):
    torch.manual_seed(b)
    cache_indices = _cache_indices(b, torch.int64)
    ref = _ref_verify(b, draft_token_num, cache_indices)
    got = fused_extend_sconv_metadata(
        B=b,
        T=b * draft_token_num,
        cache_indices=cache_indices,
        his_mode=HIS_ONES,
        draft_token_num=draft_token_num,
    )
    assert got is not None
    _assert_equal(got, ref)


@requires_cuda
def test_cu_not_spanning_T():
    """Dummy capture sequences: cu stops short of T; trailing si rows clamp to
    B-1 exactly like the reference's searchsorted + clamp."""
    b = 5
    lens = torch.tensor([3, 0, 4, 0, 2], dtype=torch.int64, device="cuda")
    T = int(lens.sum().item()) + 17
    cache_indices = _cache_indices(b, torch.int32)
    seq_lens = lens + 1
    ref = _ref_extend(b, lens, HIS_SEQ_MINUS_EXT, seq_lens, cache_indices, T)
    got = fused_extend_sconv_metadata(
        B=b,
        T=T,
        cache_indices=cache_indices,
        his_mode=HIS_SEQ_MINUS_EXT,
        extend_seq_lens=lens,
        his_src=seq_lens,
    )
    assert got is not None
    _assert_equal(got, ref)


@requires_cuda
def test_fallback_past_batch_bound():
    b = 1024  # > _FUSED_EXTEND_MAX_B
    lens = torch.ones(b, dtype=torch.int64, device="cuda")
    got = fused_extend_sconv_metadata(
        B=b,
        T=b,
        cache_indices=_cache_indices(b, torch.int32),
        his_mode=HIS_ZEROS,
        extend_seq_lens=lens,
    )
    assert got is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
