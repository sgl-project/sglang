"""Correctness tests for ``gemma4_fused_routing``.

Compares the Triton-fused routing kernel against the original SGLang
``Gemma4MoE.routing_function`` reference (softmax-of-topk * per_expert_scale).
Run with::

    pytest test/srt/layers/test_gemma4_fused_routing.py -v

Requires a CUDA-capable GPU; skips otherwise.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gemma4_fused_routing is a CUDA-only Triton kernel",
)


@pytest.fixture(scope="module")
def fused_routing():
    from sglang.srt.layers.gemma4_fused_ops import gemma4_fused_routing

    return gemma4_fused_routing


def _reference(gating_output: torch.Tensor, per_expert_scale: torch.Tensor, topk: int):
    """The previous (now fallback) torch routing function from gemma4_causal.py."""
    topk_logits, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_logits, dim=-1)
    topk_weights = topk_weights * per_expert_scale[topk_ids].to(topk_weights.dtype)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("T", [1, 7, 64, 128, 1024])
@pytest.mark.parametrize("E,K", [(128, 8), (64, 4), (256, 8)])
def test_matches_reference(fused_routing, dtype, T, E, K):
    torch.manual_seed(0)
    g = torch.randn(T, E, dtype=dtype, device="cuda")
    s = torch.rand(E, dtype=dtype, device="cuda") * 2.0

    ref_w, ref_i = _reference(g, s, K)
    out_w, out_i = fused_routing(g, s, K)

    assert out_w.dtype == torch.float32
    assert out_i.dtype == torch.int32
    assert out_w.shape == (T, K)
    assert out_i.shape == (T, K)

    # IDs must match exactly (top-K with stable tie-breaking on expert id).
    # In practice with random logits ties almost never happen; if they do we
    # accept either order as long as the weight sum and the selected set are
    # equivalent.
    # The fused kernel does softmax in fp32 throughout, while the torch
    # fallback runs softmax in the input dtype before casting to fp32.  For
    # bf16 inputs that means our kernel is *more* accurate; loosen the
    # tolerance to roughly the input-dtype eps so we don't false-fail.
    if dtype == torch.bfloat16:
        atol, rtol = 5e-3, 5e-3
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-5, 1e-5

    if (out_i != ref_i).any():
        # Compare as sets per row.
        ref_set = ref_i.sort(dim=-1).values
        out_set = out_i.sort(dim=-1).values
        assert torch.equal(
            out_set, ref_set
        ), "fused routing picked a different top-K set than reference"
        # Sum of weights per row should still be close (softmax over the same
        # K logits).
        torch.testing.assert_close(
            out_w.sum(dim=-1).to(torch.float32),
            ref_w.sum(dim=-1).to(torch.float32),
            atol=atol,
            rtol=rtol,
        )
    else:
        # Same IDs in the same order — weights must match within input dtype eps.
        torch.testing.assert_close(out_w, ref_w, atol=atol, rtol=rtol)


def test_zero_tokens(fused_routing):
    g = torch.empty(0, 128, dtype=torch.bfloat16, device="cuda")
    s = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    w, i = fused_routing(g, s, 8)
    assert w.shape == (0, 8) and i.shape == (0, 8)
    assert w.dtype == torch.float32 and i.dtype == torch.int32


def test_scale_applied(fused_routing):
    """Weights must include per_expert_scale[topk_ids]."""
    torch.manual_seed(1)
    T, E, K = 4, 128, 8
    g = torch.randn(T, E, dtype=torch.bfloat16, device="cuda")
    s = torch.rand(E, dtype=torch.bfloat16, device="cuda") * 3.0

    out_w, out_i = fused_routing(g, s, K)
    ref_w, ref_i = _reference(g, s, K)
    torch.testing.assert_close(out_w, ref_w, atol=5e-3, rtol=5e-3)
    assert torch.equal(out_i, ref_i)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
