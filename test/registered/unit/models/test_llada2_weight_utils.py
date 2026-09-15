import unittest

import torch

from sglang.srt.model_loader.llada2_weight_utils import prepare_llada2_language_weights
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

    def test_rejects_invalid_fused_expert_shape(self):
        with self.assertRaisesRegex(ValueError, "expected first dimension 2"):
            list(
                prepare_llada2_language_weights(
                    [
                        (
                            "model.language_model.layers.1.mlp.experts.down_proj",
                            torch.empty(3, 4, 5),
                        )
                    ],
                    num_experts=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
