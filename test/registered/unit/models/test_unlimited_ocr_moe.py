"""Expert checkpoint dispatch and packing oracle for Unlimited-OCR."""

import unittest

import torch

from sglang.srt.models.unlimited_ocr_moe import (
    load_unlimited_ocr_expert_weight,
    resolve_expert_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestUnlimitedOCRMoE(CustomTestCase):
    def test_expert_mapping_and_shared_bypass(self):
        prefix = "model.model.layers.2.mlp."
        for projection, target, shard in (
            ("gate_proj", "w13_weight", "w1"),
            ("up_proj", "w13_weight", "w3"),
            ("down_proj", "w2_weight", "w2"),
        ):
            self.assertEqual(
                resolve_expert_weight(f"{prefix}experts.63.{projection}.weight"),
                (f"{prefix}experts.{target}", 63, shard),
            )
        for name in ("shared_experts.gate_proj.weight", "gate.weight"):
            self.assertIsNone(resolve_expert_weight(prefix + name))

    def test_loader_dispatch_preserves_expert_identity_and_numeric_output(self):
        torch.manual_seed(31)
        experts, hidden, intermediate, tokens = 4, 6, 8, 3
        gate = torch.randn(experts, intermediate, hidden)
        up = torch.randn_like(gate)
        down = torch.randn(experts, hidden, intermediate)
        x = torch.randn(tokens, hidden)
        prefix = "model.model.layers.1.mlp.experts."
        w13 = torch.nn.Parameter(torch.zeros(experts, 2 * intermediate, hidden))
        w2 = torch.nn.Parameter(torch.zeros_like(down))

        # Boundary substitute for the external FusedMoE weight loader. Dispatch
        # must deliver the right projection, expert and parameter; oracle below
        # computes from original checkpoint tensors without using the mapper.
        def loader(param, source, target, shard_id, expert_id):
            with torch.no_grad():
                if shard_id == "w1":
                    param[expert_id, :intermediate].copy_(source)
                elif shard_id == "w3":
                    param[expert_id, intermediate:].copy_(source)
                elif shard_id == "w2":
                    param[expert_id].copy_(source)
                else:
                    raise ValueError(shard_id)

        w13.weight_loader = w2.weight_loader = loader
        params = {prefix + "w13_weight": w13, prefix + "w2_weight": w2}
        for expert in reversed(range(experts)):
            for projection, raw in (
                ("up_proj", up),
                ("down_proj", down),
                ("gate_proj", gate),
            ):
                result = load_unlimited_ocr_expert_weight(
                    params, f"{prefix}{expert}.{projection}.weight", raw[expert]
                )
                self.assertIn(result, params)
        ids = torch.tensor([[3, 0], [1, 2], [0, 1]])
        weights = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        expected, actual = torch.zeros_like(x), torch.zeros_like(x)
        for token in range(tokens):
            for choice in range(2):
                expert = int(ids[token, choice])
                expected[token] += (
                    weights[token, choice]
                    * (
                        torch.nn.functional.silu(x[token] @ gate[expert].T)
                        * (x[token] @ up[expert].T)
                    )
                    @ down[expert].T
                )
                packed = x[token] @ w13[expert].T
                actual[token] += (
                    weights[token, choice]
                    * (
                        torch.nn.functional.silu(packed[:intermediate])
                        * packed[intermediate:]
                    )
                    @ w2[expert].T
                )
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(
            load_unlimited_ocr_expert_weight({}, prefix + "0.gate_proj.weight", gate[0])
        )


if __name__ == "__main__":
    unittest.main()
