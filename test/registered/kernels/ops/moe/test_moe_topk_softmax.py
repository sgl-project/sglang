"""Correctness tests for the JIT :func:`topk_softmax` MoE router.

The JIT kernel is a host-side port of the AOT ``sgl_kernel.topk_softmax``: the
device code is unchanged, only the host launcher moves to the tvm-ffi
``TensorView`` API and the softmax workspace is allocated by the Python wrapper.

We validate it two ways:

* against a definition-based torch reference (documents the math), and
* against the AOT kernel it replaces, when ``sgl_kernel`` is importable.

Both expert-count regimes are covered: the warp-specialized fast path
(power-of-two ``num_experts`` <= 512, no scratch) and the two-pass
softmax + top-k path that everything else falls back to. 512 and 1024 are both
in the matrix so the boundary between them is pinned on either side.

Against the torch reference, index comparisons are tie-robust: rather than
requiring identical index tensors, we check that the probability sitting at each
returned index matches the returned weight, so an arbitrary but valid tie-break
is accepted. Against the AOT kernel the comparison is exact, because the device
code is the same on both sides.
"""

from __future__ import annotations

import sys
from typing import Optional

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe.moe_topk_softmax import topk_softmax
from sglang.test.ci.ci_register import register_cuda_ci

# CI runs the trimmed matrix (17 cases, one dtype), but on a cold runner the
# single JIT compile is ~29s of the ~31s total -- the case count is nearly free.
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"

try:
    from sgl_kernel import topk_softmax as aot_topk_softmax

    AOT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the installed wheel
    aot_topk_softmax = None
    AOT_AVAILABLE = False

DTYPES = get_ci_test_range(
    full_range=[torch.float32, torch.float16, torch.bfloat16],
    ci_range=[torch.bfloat16],
)
# 8/128/256/512 exercise the warp-specialized power-of-two fast path;
# 6/160/1024 exercise the workspace (two-pass) path. 512 and 1024 sit either
# side of the boundary between them, so both must stay in the list.
NUM_EXPERTS = get_ci_test_range(
    full_range=[8, 128, 256, 512, 6, 160, 1024],
    ci_range=[8, 160],
)
TOPKS = get_ci_test_range(full_range=[1, 2, 4, 8], ci_range=[2])
SOFTCAPS = get_ci_test_range(full_range=[0.0, 30.0], ci_range=[0.0])


def _reference_probs(
    gating_output: torch.Tensor,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Definition-based reference for the routing probabilities."""
    logits = gating_output.float()
    if moe_softcapping:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    return torch.softmax(logits, dim=-1)


def _run_jit(gating_output: torch.Tensor, topk: int, renormalize: bool, softcap, bias):
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    topk_softmax(topk_weights, topk_ids, gating_output, renormalize, softcap, bias)
    return topk_weights, topk_ids


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("moe_softcapping", SOFTCAPS)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax_vs_torch(
    dtype, num_experts, topk, moe_softcapping, use_bias, renormalize
):
    if topk > num_experts:
        pytest.skip("topk must be <= num_experts")

    num_tokens = 200
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device=DEVICE)
    correction_bias = (
        torch.randn(num_experts, dtype=torch.float32, device=DEVICE)
        if use_bias
        else None
    )

    weights, ids = _run_jit(
        gating_output, topk, renormalize, moe_softcapping, correction_bias
    )

    probs = _reference_probs(gating_output, moe_softcapping, correction_bias)
    ref_weights, _ = probs.topk(topk, dim=-1)
    expected = (
        ref_weights / ref_weights.sum(-1, keepdim=True) if renormalize else ref_weights
    )

    tol = 1e-3 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(weights, expected, rtol=tol, atol=tol)

    # Tie-robust index check: the probability at each returned index must equal
    # the returned weight (undoing renormalization first).
    gathered = torch.gather(probs, 1, ids.long())
    unnormalized = (
        weights * ref_weights.sum(-1, keepdim=True) if renormalize else weights
    )
    torch.testing.assert_close(gathered, unnormalized, rtol=tol, atol=tol)

    # Indices must be distinct within a row.
    sorted_ids, _ = ids.sort(dim=-1)
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all(), "duplicate expert ids"
    assert ((ids >= 0) & (ids < num_experts)).all(), "expert id out of range"


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel (AOT) is not importable")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax_matches_aot(dtype, num_experts, topk, renormalize):
    """The JIT port must be numerically identical to the AOT kernel."""
    if topk > num_experts:
        pytest.skip("topk must be <= num_experts")

    num_tokens = 512
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device=DEVICE)

    jit_weights, jit_ids = _run_jit(gating_output, topk, renormalize, 0.0, None)

    aot_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=DEVICE)
    aot_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=DEVICE)
    aot_topk_softmax(aot_weights, aot_ids, gating_output, renormalize, 0.0, None)

    # The device code is unchanged from the AOT kernel and both dispatchers
    # agree on which path each expert count takes, so this is bit-identical --
    # on the warp-specialized path and on the two-pass path alike. Keeping the
    # tolerance at exactly zero is deliberate: it is what caught the dispatcher
    # falling out of sync with the AOT one at num_experts == 512.
    assert torch.equal(jit_ids, aot_ids)
    torch.testing.assert_close(jit_weights, aot_weights, rtol=0, atol=0)


@pytest.mark.parametrize("num_experts", [8, 160])
def test_topk_softmax_single_token(num_experts):
    gating_output = torch.randn((1, num_experts), dtype=torch.bfloat16, device=DEVICE)
    weights, ids = _run_jit(gating_output, 2, True, 0.0, None)
    torch.testing.assert_close(
        weights.sum(-1), torch.ones(1, device=DEVICE), rtol=1e-2, atol=1e-2
    )
    assert ids.shape == (1, 2)


@pytest.mark.parametrize("num_experts", [8, 160])
def test_topk_softmax_full_topk(num_experts):
    """topk == num_experts: weights must be a permutation of the full softmax."""
    gating_output = torch.randn((16, num_experts), dtype=torch.float32, device=DEVICE)
    weights, ids = _run_jit(gating_output, num_experts, False, 0.0, None)
    probs = _reference_probs(gating_output, 0.0, None)
    torch.testing.assert_close(
        weights.sort(dim=-1).values, probs.sort(dim=-1).values, rtol=1e-3, atol=1e-3
    )
    assert (
        ids.sort(dim=-1)
        .values.eq(torch.arange(num_experts, device=DEVICE, dtype=torch.int32))
        .all()
    )


def test_topk_softmax_zero_tokens():
    """An empty batch must be a no-op rather than a launch failure."""
    gating_output = torch.randn((0, 8), dtype=torch.bfloat16, device=DEVICE)
    weights, ids = _run_jit(gating_output, 2, False, 0.0, None)
    assert weights.shape == (0, 2) and ids.shape == (0, 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
