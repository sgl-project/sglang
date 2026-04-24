"""Correctness tests for the fused topk+pack kernels used by
``flashinfer_trtllm_routed``.

Covers the two Triton kernels in ``sglang.srt.layers.moe.topk`` that together
replace the old ``_pack_topk_for_flashinfer_routed`` op chain:

* ``fused_topk_softmax_pack_flashinfer_triton`` — fuses raw-logits topk,
  optional softmax renormalize, bf16 cast, and FlashInfer pack.
* ``pack_topk_for_flashinfer_triton`` — standalone Triton pack used when a
  non-fused topk path (e.g. biased/grouped topk) produces
  ``StandardTopKOutput`` that still has to reach the packed layout.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.topk import (
    fused_topk_softmax_pack_flashinfer_triton,
    pack_topk_for_flashinfer_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-large")


def _torch_pack(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Reference pack used before the Triton kernels landed."""
    packed_ids = topk_ids.to(torch.int32)
    packed_weights = topk_weights.to(torch.bfloat16)
    return (packed_ids << 16) | packed_weights.view(torch.int16).to(torch.int32)


def _torch_topk_softmax_pack(
    gating: torch.Tensor, topk: int, renormalize: bool
) -> torch.Tensor:
    """Reference topk + optional softmax + pack (the op chain being fused)."""
    _, topk_ids = torch.topk(gating, k=topk, dim=-1, sorted=False)
    topk_weights = gating.float().gather(1, topk_ids)
    if renormalize:
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
    return _torch_pack(topk_ids.to(torch.int32), topk_weights.to(torch.float32))


def _decode(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover (ids, weights) from FlashInfer-packed int32 for comparison.

    Assumes non-negative weights, which is always true for softmax outputs and
    for the scenarios exercised here (the FlashInfer layout relies on this).
    """
    as_i64 = packed.to(torch.int64)
    ids = ((as_i64 >> 16) & 0xFFFF).to(torch.int32)
    low_i16 = (as_i64 & 0xFFFF).to(torch.int32).to(torch.int16)
    weights = low_i16.view(torch.bfloat16).to(torch.float32)
    return ids, weights


def _assert_packed_rows_match(got: torch.Tensor, expected: torch.Tensor) -> None:
    """Check per-row (id, weight) multiset equality.

    ``torch.topk(sorted=False)`` and the Triton kernel's greedy max-selection
    order are both unspecified within a row, so we compare as sets.
    """
    assert got.shape == expected.shape
    assert got.dtype == torch.int32 == expected.dtype
    if got.numel() == 0:
        return
    ids_g, w_g = _decode(got)
    ids_e, w_e = _decode(expected)
    sorted_ids_g, perm_g = ids_g.sort(dim=-1)
    sorted_ids_e, perm_e = ids_e.sort(dim=-1)
    assert torch.equal(sorted_ids_g, sorted_ids_e)
    w_g_sorted = torch.gather(w_g, 1, perm_g)
    w_e_sorted = torch.gather(w_e, 1, perm_e)
    # bf16 rounding is the only expected source of drift.
    torch.testing.assert_close(w_g_sorted, w_e_sorted, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 2),
        (7, 32, 3),  # non-power-of-2 topk
        (16, 64, 4),
        (32, 32, 1),
        (128, 128, 8),
        (1024, 128, 8),
        (3, 256, 8),
        (4, 7, 5),  # non-power-of-2 experts
        (0, 128, 8),  # empty batch
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fused_topk_softmax_pack_flashinfer(shape, renormalize, dtype):
    """Fused kernel must match raw-logits topk + softmax + torch pack."""
    num_tokens, num_experts, topk = shape
    torch.manual_seed(num_tokens * 131 + num_experts * 7 + topk)

    gating = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)

    got = fused_topk_softmax_pack_flashinfer_triton(gating, topk, renormalize)
    expected = _torch_topk_softmax_pack(gating, topk, renormalize)

    _assert_packed_rows_match(got, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (16, 4),
        (128, 8),
        (7, 3),
        (1024, 8),
        (4, 5),
        (0, 8),  # empty batch
    ],
)
@pytest.mark.parametrize(
    "ids_dtype", [torch.int32, torch.int64]
)  # downstream paths use both
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
def test_pack_topk_for_flashinfer(shape, ids_dtype, weights_dtype):
    """Pack-only Triton kernel must match the original torch op chain."""
    num_tokens, topk = shape
    torch.manual_seed(num_tokens * 257 + topk)

    if num_tokens == 0:
        ids = torch.empty(0, topk, dtype=ids_dtype, device="cuda")
        weights = torch.empty(0, topk, dtype=weights_dtype, device="cuda")
    else:
        # Use enough experts that ids cover a realistic range.
        ids = torch.randint(0, 128, (num_tokens, topk), device="cuda", dtype=ids_dtype)
        # Non-negative, in [0, 1] — matches softmax output semantics.
        weights = torch.rand(num_tokens, topk, device="cuda", dtype=weights_dtype)

    got = pack_topk_for_flashinfer_triton(ids, weights)
    expected = _torch_pack(ids, weights)

    assert got.shape == expected.shape == (num_tokens, topk)
    assert got.dtype == torch.int32
    if num_tokens > 0:
        # Pack is a pure bit-manipulation with no reordering — equality is exact
        # up to bf16 rounding of the weights, which both paths apply.
        assert torch.equal(got, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
