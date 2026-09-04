import unittest

import torch

from sglang.srt.models.llada2_weight_utils import prepare_llada2_language_weights
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLLaDA2WeightUtils(CustomTestCase):
    def test_prepares_clean_language_checkpoint_layout(self):
        fused = torch.arange(24).reshape(2, 3, 4)
        lm_head = torch.randn(4, 3)

        expanded = list(
            prepare_llada2_language_weights(
                [
                    ("model.language_model.layers.1.mlp.experts.gate_proj", fused),
                    ("model.lm_head.weight", lm_head),
                ],
                num_experts=2,
            )
        )

        self.assertEqual(
            [name for name, _ in expanded],
            [
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.1.mlp.experts.1.gate_proj.weight",
                "lm_head.weight",
            ],
        )
        torch.testing.assert_close(expanded[0][1], fused[0])
        torch.testing.assert_close(expanded[1][1], fused[1])
        torch.testing.assert_close(expanded[2][1], lm_head)

    def test_expert_scale_expansion_is_opt_in(self):
        gate_scale = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        up_scale = gate_scale + 100
        down_scale = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        weights = [
            (
                "model.language_model.layers.1.mlp.experts.gate_proj_scale",
                gate_scale,
            ),
            (
                "model.language_model.layers.1.mlp.experts.up_proj_scale",
                up_scale,
            ),
            (
                "model.language_model.layers.1.mlp.experts.down_proj_scale",
                down_scale,
            ),
        ]

        unexpanded = list(prepare_llada2_language_weights(weights, num_experts=2))
        self.assertEqual(
            [name for name, _ in unexpanded],
            [
                "model.layers.1.mlp.experts.gate_proj_scale",
                "model.layers.1.mlp.experts.up_proj_scale",
                "model.layers.1.mlp.experts.down_proj_scale",
            ],
        )

        expanded = list(
            prepare_llada2_language_weights(
                weights,
                num_experts=2,
                expand_expert_scales=True,
            )
        )

        self.assertEqual(
            [name for name, _ in expanded],
            [
                "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv",
                "model.layers.1.mlp.experts.1.gate_proj.weight_scale_inv",
                "model.layers.1.mlp.experts.0.up_proj.weight_scale_inv",
                "model.layers.1.mlp.experts.1.up_proj.weight_scale_inv",
                "model.layers.1.mlp.experts.0.down_proj.weight_scale_inv",
                "model.layers.1.mlp.experts.1.down_proj.weight_scale_inv",
            ],
        )
        torch.testing.assert_close(expanded[0][1], gate_scale[0])
        torch.testing.assert_close(expanded[3][1], up_scale[1])
        torch.testing.assert_close(expanded[5][1], down_scale[1])

    def test_rejects_invalid_fused_expert_shape(self):
        cases = [
            (
                "weight",
                "model.language_model.layers.1.mlp.experts.down_proj",
                False,
            ),
            (
                "scale",
                "model.language_model.layers.1.mlp.experts.gate_proj_scale",
                True,
            ),
        ]
        for label, name, expand_expert_scales in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "expected first dimension 2"):
                    list(
                        prepare_llada2_language_weights(
                            [(name, torch.empty(3, 4, 5))],
                            num_experts=2,
                            expand_expert_scales=expand_expert_scales,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
