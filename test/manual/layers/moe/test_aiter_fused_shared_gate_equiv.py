"""GPU equivalence test for the aiter fused-shared-gate wrapper.

Manual (not CI-registered) -- needs an AMD GPU and a rebuilt aiter that ships
`topk_softmax_fused_shared_gate`; it skips cleanly otherwise. Run on the server:

    python3 -m pytest test/manual/layers/moe/test_aiter_fused_shared_gate_equiv.py -v

Validates `aiter_fused_softmax_topk_with_shared_gate` end-to-end (real output
buffers + the real aiter kernel) against the reference the legacy routed-topk +
shared-append path computes: routed softmax top-k (+ optional renorm) for the
routed columns, and sigmoid(hidden @ gate.T) * scale (scale AFTER sigmoid) for
the shared columns written at ids base + s. This closes the sglang-layer gap the
CPU contract test cannot cover (that one mocks the kernel).
"""

import unittest

import torch

from sglang.srt.layers.moe import topk as topk_module


def _aiter_ready():
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(topk_module, "aiter_topk_softmax_fused_shared_gate", None)
        is not None
        and getattr(topk_module, "_use_aiter", False)
    )


def _reference(gating, hidden, gate_weight, topk, num_shared, base, scale, renorm):
    probs = torch.softmax(gating.float(), dim=-1)
    routed_w, routed_i = torch.topk(probs, topk, dim=-1)
    if renorm:
        routed_w = routed_w / routed_w.sum(dim=-1, keepdim=True)
    shared_w = torch.sigmoid(hidden.float() @ gate_weight.float().t()) * scale
    m = gating.shape[0]
    shared_i = (
        (base + torch.arange(num_shared, device=gating.device, dtype=torch.int32))
        .unsqueeze(0)
        .expand(m, num_shared)
    )
    return routed_w, routed_i.to(torch.int32), shared_w, shared_i


def _sorted_pairs(ids, weights):
    order = torch.argsort(ids, dim=-1)
    return torch.gather(ids, -1, order), torch.gather(weights, -1, order)


@unittest.skipUnless(
    _aiter_ready(), "needs AMD GPU + aiter with topk_softmax_fused_shared_gate"
)
class TestAiterFusedSharedGateEquiv(unittest.TestCase):
    def _run(self, tokens, num_experts, hidden, topk, num_shared, scale, renorm):
        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        base = num_experts
        gating = torch.randn(tokens, num_experts, device=dev, dtype=dt)
        hs = torch.randn(tokens, hidden, device=dev, dtype=dt) * 0.1
        gate_weight = torch.randn(num_shared, hidden, device=dev, dtype=dt) * 0.02

        weights, ids = topk_module.aiter_fused_softmax_topk_with_shared_gate(
            hs,
            gating,
            top_k=topk,
            num_fused_shared_experts=num_shared,
            shared_expert_base=base,
            gate_weight=gate_weight,
            renormalize=renorm,
            shared_expert_scale=scale,
        )

        ref_rw, ref_ri, ref_sw, ref_si = _reference(
            gating, hs, gate_weight, topk, num_shared, base, scale, renorm
        )
        got_rw, got_ri = weights[:, :topk], ids[:, :topk]
        got_sw, got_si = weights[:, topk:], ids[:, topk:]

        # Shared columns: ids exact, weights within tol.
        torch.testing.assert_close(got_si.to(torch.int32), ref_si, rtol=0, atol=0)
        torch.testing.assert_close(got_sw.float(), ref_sw, rtol=2e-2, atol=2e-2)

        # Routed columns: compare as per-row sets (kernel/topk order may differ).
        ref_i, ref_w = _sorted_pairs(ref_ri, ref_rw)
        g_i, g_w = _sorted_pairs(got_ri.to(torch.int32), got_rw.float())
        torch.testing.assert_close(g_i, ref_i, rtol=0, atol=0)
        torch.testing.assert_close(g_w, ref_w, rtol=2e-2, atol=2e-2)

    def test_qwen_shape(self):
        # Qwen3.5: 512 experts, top_k=10, 1 shared expert, hidden=4096, bf16.
        self._run(64, 512, 4096, 10, 1, 1.0, True)

    def test_variants(self):
        for tokens in (1, 17):
            for scale in (1.0, 0.5):
                for renorm in (True, False):
                    self._run(tokens, 512, 4096, 10, 1, scale, renorm)


if __name__ == "__main__":
    unittest.main()
