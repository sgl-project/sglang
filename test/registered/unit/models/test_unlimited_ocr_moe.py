"""Expert checkpoint dispatch and packing for Unlimited-OCR."""

import unittest

import torch

from sglang.srt.models.unlimited_ocr_moe import (
    load_unlimited_ocr_expert_weight,
    resolve_expert_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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
        # The fused loader owns plain weights only; scales, biases, shared
        # experts and the router must not be captured by this mapping.
        for name in (
            "shared_experts.gate_proj.weight",
            "gate.weight",
            "experts.0.gate_proj.weight_scale",
            "experts.0.gate_proj.bias",
        ):
            self.assertIsNone(resolve_expert_weight(prefix + name))

    def test_loader_packs_projections_into_fused_layout(self):
        experts, hidden, intermediate = 4, 6, 8
        gate = torch.arange(experts * intermediate * hidden).reshape(
            experts, intermediate, hidden
        )
        up = gate + 1000
        down = gate + 2000
        prefix = "model.model.layers.1.mlp.experts."
        w13 = torch.nn.Parameter(torch.zeros(experts, 2 * intermediate, hidden))
        w2 = torch.nn.Parameter(torch.zeros_like(down))

        # Boundary substitute for the external FusedMoE weight loader: the
        # packing contract under test is which slice each projection lands in.
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

        torch.testing.assert_close(w13[:, :intermediate], gate)
        torch.testing.assert_close(w13[:, intermediate:], up)
        torch.testing.assert_close(w2, down)
        self.assertIsNone(
            load_unlimited_ocr_expert_weight({}, prefix + "0.gate_proj.weight", gate[0])
        )


if __name__ == "__main__":
    unittest.main()
