"""Unit test for the aiter Option-A fused-shared-gate topk wrapper.

`aiter_fused_softmax_topk_with_shared_gate` replaces the routed-topk +
`_append_shared_to_topk_output` pair on the aiter path (see qwen2_moe). The
fused-kernel *numerics* are covered by aiter's op_test
(op_tests/test_topk_softmax_shared_gate.py); what is only testable here -- and
where a silent regression would hide -- is the wrapper's contract with the
kernel: output-buffer width/dtype and faithful forwarding of every shared-gate
argument. This test mocks the kernel and pins that contract on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.test.test_utils import CustomTestCase


class TestAiterFusedSoftmaxTopkSharedGate(CustomTestCase):
    M = 3
    HIDDEN = 16
    NUM_EXPERTS = 512
    TOP_K = 8
    NUM_SHARED = 2
    SCALE = 0.5

    def _call(self, renormalize):
        torch.manual_seed(0)
        hidden_states = torch.randn(self.M, self.HIDDEN)
        router_logits = torch.randn(self.M, self.NUM_EXPERTS)
        gate_weight = torch.randn(self.NUM_SHARED, self.HIDDEN)
        base = self.NUM_EXPERTS
        total = self.TOP_K + self.NUM_SHARED

        captured = {}

        def fake_topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            need_renorm,
            **kwargs,
        ):
            captured["weights_buf"] = topk_weights
            captured["ids_buf"] = topk_ids
            captured["tei_buf"] = token_expert_indices
            captured["gating_output"] = gating_output
            captured["need_renorm"] = need_renorm
            captured["kwargs"] = kwargs
            # Emulate the kernel writing into the caller-owned buffers so we can
            # confirm the wrapper returns the same buffers it allocated.
            topk_weights.fill_(0.25)
            topk_ids.copy_(torch.arange(total, dtype=torch.int32).expand(self.M, total))

        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(
                topk_module, "aiter_topk_softmax", fake_topk_softmax, create=True
            ),
        ):
            weights, ids = topk_module.aiter_fused_softmax_topk_with_shared_gate(
                hidden_states,
                router_logits,
                top_k=self.TOP_K,
                num_fused_shared_experts=self.NUM_SHARED,
                shared_expert_base=base,
                gate_weight=gate_weight,
                renormalize=renormalize,
                shared_expert_scale=self.SCALE,
            )
        return captured, weights, ids, hidden_states, router_logits, gate_weight, base

    def test_buffer_shapes_and_dtypes(self):
        captured, weights, ids, *_ = self._call(renormalize=True)
        total = self.TOP_K + self.NUM_SHARED
        for buf, dtype in (
            (captured["weights_buf"], torch.float32),
            (captured["ids_buf"], torch.int32),
            (captured["tei_buf"], torch.int32),
        ):
            self.assertEqual(tuple(buf.shape), (self.M, total))
            self.assertEqual(buf.dtype, dtype)
        # The wrapper must return the very buffers the kernel filled, unmodified.
        self.assertIs(weights, captured["weights_buf"])
        self.assertIs(ids, captured["ids_buf"])
        self.assertTrue(torch.all(weights == 0.25))

    def test_shared_gate_args_forwarded(self):
        (
            captured,
            _weights,
            _ids,
            hidden_states,
            router_logits,
            gate_weight,
            base,
        ) = self._call(renormalize=False)

        # Routed inputs forwarded verbatim (same tensor objects).
        self.assertIs(captured["gating_output"], router_logits)
        self.assertIs(captured["kwargs"]["hidden_states"], hidden_states)
        self.assertIs(captured["kwargs"]["gate_weight"], gate_weight)
        # renormalize threads through to the kernel's need_renorm flag.
        self.assertFalse(captured["need_renorm"])
        # Shared-gate parameters must match the wrapper's arguments exactly.
        self.assertEqual(captured["kwargs"]["num_shared_experts"], self.NUM_SHARED)
        self.assertEqual(captured["kwargs"]["shared_expert_scoring_func"], "sigmoid")
        self.assertEqual(captured["kwargs"]["shared_expert_base"], base)
        self.assertEqual(captured["kwargs"]["shared_expert_scale"], self.SCALE)

    def test_requires_use_aiter(self):
        with patch.object(topk_module, "_use_aiter", False):
            with self.assertRaises(AssertionError):
                topk_module.aiter_fused_softmax_topk_with_shared_gate(
                    torch.randn(self.M, self.HIDDEN),
                    torch.randn(self.M, self.NUM_EXPERTS),
                    top_k=self.TOP_K,
                    num_fused_shared_experts=self.NUM_SHARED,
                    shared_expert_base=self.NUM_EXPERTS,
                    gate_weight=torch.randn(self.NUM_SHARED, self.HIDDEN),
                    renormalize=True,
                    shared_expert_scale=self.SCALE,
                )


if __name__ == "__main__":
    unittest.main()
