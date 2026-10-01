"""SGLANG_OPT_AITER_FUSED_GATE_SHARED: on the aiter path, DeepSeek-V4's sqrtsoftplus
gate emits the fused shared-expert slot itself.

Pins that it routes like the default path (aiter topk_gating + the shared-expert
append), that the shared expert appears exactly once with weight 1.0, and that the
routed weights carry routed_scaling_factor, as the aiter MoE runner expects.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="jit-kernel-unit-test-amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

E, TOPK_ROUTED, SHARED, SCALE = 384, 6, 1, 2.5


def _select(monkeypatch, fused, logits, bias):
    monkeypatch.setenv("SGLANG_OPT_AITER_FUSED_GATE_SHARED", "1" if fused else "0")
    if fused:
        import aiter

        def _unexpected(*args, **kwargs):
            raise AssertionError("aiter topk_gating ran with the fused gate enabled")

        monkeypatch.setattr(aiter, "topk_gating", _unexpected)
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    cfg = TopKConfig(
        top_k=TOPK_ROUTED + SHARED,
        renormalize=True,
        num_fused_shared_experts=SHARED,
        scoring_func="sqrtsoftplus",
        correction_bias=bias,
        routed_scaling_factor=SCALE,
    )
    hidden = torch.empty(logits.shape[0], 16, dtype=torch.bfloat16, device="cuda")
    out = select_experts(hidden, logits, cfg)
    return out.topk_ids.long(), out.topk_weights.float()


def _inputs(tokens):
    torch.manual_seed(0)
    bias = (0.1 * torch.randn(E, device="cuda")).to(torch.bfloat16)
    logits = torch.randn(tokens, E, dtype=torch.bfloat16, device="cuda")
    return logits, bias


def _aiter_only():
    from sglang.srt.layers.moe import topk as topk_mod

    if not topk_mod._use_aiter:
        pytest.skip("the fused shared slot is an aiter-path option")


@pytest.mark.parametrize("tokens", [1, 7, 98, 448, 2048])
def test_fused_gate_routes_like_the_default_path(monkeypatch, tokens):
    _aiter_only()
    logits, bias = _inputs(tokens)
    ref_ids, ref_w = _select(monkeypatch, False, logits, bias)
    ids, w = _select(monkeypatch, True, logits, bias)

    assert ids.shape == ref_ids.shape == (tokens, TOPK_ROUTED + SHARED)
    # Same experts per row (order may differ), each with the same weight. Exact
    # score ties may break either way; nothing else may.
    ref_order, order = ref_ids.argsort(-1), ids.argsort(-1)
    ref_sorted, got_sorted = ref_ids.gather(-1, ref_order), ids.gather(-1, order)
    same = (ref_sorted == got_sorted).all(-1)
    score = torch.sqrt(torch.nn.functional.softplus(logits.float())) + bias.float()
    for r in (~same).nonzero().flatten().tolist():
        only_ref = sorted(set(ref_sorted[r].tolist()) - set(got_sorted[r].tolist()))
        only_got = sorted(set(got_sorted[r].tolist()) - set(ref_sorted[r].tolist()))
        for x, y in zip(only_ref, only_got):
            assert score[r, x].item() == score[r, y].item(), f"row {r}: {x} vs {y}"
    torch.testing.assert_close(
        w.gather(-1, order)[same],
        ref_w.gather(-1, ref_order)[same],
        rtol=1e-5,
        atol=1e-6,
    )


def test_shared_expert_appears_once_with_unit_weight(monkeypatch):
    _aiter_only()
    logits, bias = _inputs(64)
    ids, w = _select(monkeypatch, True, logits, bias)

    shared = ids >= E
    assert torch.equal(shared.sum(-1), torch.full_like(shared.sum(-1), SHARED))
    assert (ids[shared] == E).all()
    torch.testing.assert_close(w[shared], torch.ones_like(w[shared]))
    # Routed weights are renormalized and pre-scaled by routed_scaling_factor.
    routed_sum = w.masked_fill(shared, 0).sum(-1)
    torch.testing.assert_close(routed_sum, torch.full_like(routed_sum, SCALE))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
