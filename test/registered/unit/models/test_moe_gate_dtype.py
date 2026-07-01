"""Regression tests for BF16 MoE routing-bias initialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.srt.models.glm4_moe import Glm4MoeGate
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoeGateDtype(CustomTestCase):
    def test_flashinfer_trtllm_correction_bias_is_bf16(self):
        with patch.object(
            moe_utils, "MOE_RUNNER_BACKEND", MoeRunnerBackend.FLASHINFER_TRTLLM
        ):
            deepseek_gate = MoEGate(
                SimpleNamespace(
                    topk_method="noaux_tc",
                    n_routed_experts=8,
                    hidden_size=16,
                ),
                quant_config=None,
            )
            glm4_gate = Glm4MoeGate(SimpleNamespace(n_routed_experts=8, hidden_size=16))
            glm4_lite_gate = Glm4MoeLiteGate(
                SimpleNamespace(n_routed_experts=8, hidden_size=16)
            )

        self.assertEqual(deepseek_gate.e_score_correction_bias.dtype, torch.bfloat16)
        self.assertEqual(glm4_gate.e_score_correction_bias.dtype, torch.bfloat16)
        self.assertEqual(glm4_lite_gate.e_score_correction_bias.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
