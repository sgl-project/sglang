"""fused_decode_sconv_metadata must be bit-identical to the unfused prep.

The unfused reference is the exact op sequence `_prepare_decode_sconv_metadata`
used to launch: two arange calls + ones + precompute_helion_decode_metadata
(!= PAD, &, clamp, long, arange x2).
"""

import pytest
import torch

from sglang.srt.models.inkling_common.kernels.sconv import (
    PAD_SLOT_ID,
    fused_decode_sconv_metadata,
    precompute_helion_decode_metadata,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

# cross the BLOCK=1024 grid boundary and hit odd sizes
BATCH_SIZES = [1, 2, 3, 17, 64, 160, 257, 1023, 1024, 1025]


def _reference(B: int, cache_indices: torch.Tensor):
    device = cache_indices.device
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=device)
    has_initial_state = torch.ones(B, dtype=torch.bool, device=device)
    precomputed = precompute_helion_decode_metadata(
        B=B, W=4, cache_indices=cache_indices, has_initial_state=has_initial_state
    )
    return query_start_loc, has_initial_state, precomputed


@requires_cuda
@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_matches_unfused(b: int, idx_dtype: torch.dtype):
    torch.manual_seed(b)
    cache_indices = torch.randint(0, 4096, (b,), dtype=idx_dtype, device="cuda")
    # sprinkle PAD slots (cudagraph padding lanes)
    pad = torch.rand(b, device="cuda") < 0.25
    cache_indices[pad] = PAD_SLOT_ID

    ref_qsl, ref_his, ref_meta = _reference(b, cache_indices)
    qsl, his, meta = fused_decode_sconv_metadata(B=b, cache_indices=cache_indices)

    for tag, got, ref in (
        ("query_start_loc", qsl, ref_qsl),
        ("has_initial_state", his, ref_his),
        ("cache_mask", meta["cache_mask"], ref_meta["cache_mask"]),
        ("safe_idx", meta["safe_idx"], ref_meta["safe_idx"]),
        ("cu", meta["cu"], ref_meta["cu"]),
        ("si", meta["si"], ref_meta["si"]),
    ):
        assert got.dtype == ref.dtype, (tag, got.dtype, ref.dtype)
        assert got.shape == ref.shape, (tag, got.shape, ref.shape)
        assert torch.equal(got, ref), tag


@requires_cuda
def test_all_pad():
    cache_indices = torch.full((8,), PAD_SLOT_ID, dtype=torch.int32, device="cuda")
    _, _, meta = fused_decode_sconv_metadata(B=8, cache_indices=cache_indices)
    assert not meta["cache_mask"].any()
    assert (meta["safe_idx"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
