"""Tests for ``mask_topk_ids`` (int32 in-place) and ``mask_topk_ids_to_int64`` (fused mask+cast)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


def _ref_mask(
    topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor
) -> torch.Tensor:
    n = topk_ids.shape[0]
    cutoff = int(num_token_non_padded.item())
    out = topk_ids.clone()
    if cutoff < n:
        out[cutoff:] = -1
    return out


# ---- mask_topk_ids (int32 in-place) ----


@pytest.mark.parametrize(
    "num_tokens,topk",
    [(1, 8), (16, 8), (128, 4), (1024, 8), (4096, 16)],
)
@pytest.mark.parametrize("frac_valid", [0.0, 0.25, 0.5, 1.0])
def test_mask_topk_ids_correctness(
    num_tokens: int, topk: int, frac_valid: float
) -> None:
    from sglang.kernels.ops.attention.dsv4 import mask_topk_ids

    torch.manual_seed(0)
    topk_ids = torch.randint(
        low=0, high=2**30, size=(num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    cutoff = int(num_tokens * frac_valid)
    ntn = torch.tensor([cutoff], dtype=torch.int32, device="cuda")

    ref = _ref_mask(topk_ids, ntn)
    mask_topk_ids(topk_ids, ntn)

    assert topk_ids.dtype == torch.int32
    torch.testing.assert_close(topk_ids, ref, atol=0, rtol=0)


def test_mask_topk_ids_no_padding() -> None:
    from sglang.kernels.ops.attention.dsv4 import mask_topk_ids

    topk_ids = torch.randint(
        low=0, high=2**30, size=(64, 8), dtype=torch.int32, device="cuda"
    )
    expected = topk_ids.clone()
    mask_topk_ids(topk_ids, torch.tensor([64], dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(topk_ids, expected, atol=0, rtol=0)


def test_mask_topk_ids_all_padded() -> None:
    from sglang.kernels.ops.attention.dsv4 import mask_topk_ids

    topk_ids = torch.randint(
        low=0, high=128, size=(32, 8), dtype=torch.int32, device="cuda"
    )
    mask_topk_ids(topk_ids, torch.tensor([0], dtype=torch.int32, device="cuda"))
    expected = torch.full_like(topk_ids, -1)
    torch.testing.assert_close(topk_ids, expected, atol=0, rtol=0)


# ---- mask_topk_ids_to_int64 (fused mask + cast int32->int64) ----


@pytest.mark.parametrize(
    "num_tokens,topk",
    [(1, 8), (16, 8), (128, 4), (1024, 8), (4096, 16)],
)
@pytest.mark.parametrize("frac_valid", [0.0, 0.25, 0.5, 1.0])
def test_mask_topk_ids_to_int64_correctness(
    num_tokens: int, topk: int, frac_valid: float
) -> None:
    from sglang.kernels.ops.attention.dsv4.moe import mask_topk_ids_to_int64

    torch.manual_seed(0)
    topk_ids = torch.randint(
        low=0, high=2**30, size=(num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    cutoff = int(num_tokens * frac_valid)
    ntn = torch.tensor([cutoff], dtype=torch.int32, device="cuda")

    ref = _ref_mask(topk_ids, ntn).to(torch.int64)
    result = mask_topk_ids_to_int64(topk_ids, ntn)

    assert result.dtype == torch.int64
    assert result.shape == topk_ids.shape
    assert topk_ids.dtype == torch.int32
    torch.testing.assert_close(result, ref, atol=0, rtol=0)


def test_mask_topk_ids_to_int64_no_padding() -> None:
    from sglang.kernels.ops.attention.dsv4.moe import mask_topk_ids_to_int64

    topk_ids = torch.randint(
        low=0, high=2**30, size=(64, 8), dtype=torch.int32, device="cuda"
    )
    result = mask_topk_ids_to_int64(
        topk_ids, torch.tensor([64], dtype=torch.int32, device="cuda")
    )
    expected = topk_ids.to(torch.int64)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_mask_topk_ids_to_int64_all_padded() -> None:
    from sglang.kernels.ops.attention.dsv4.moe import mask_topk_ids_to_int64

    topk_ids = torch.randint(
        low=0, high=128, size=(32, 8), dtype=torch.int32, device="cuda"
    )
    result = mask_topk_ids_to_int64(
        topk_ids, torch.tensor([0], dtype=torch.int32, device="cuda")
    )
    expected = torch.full((32, 8), -1, dtype=torch.int64, device="cuda")
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
