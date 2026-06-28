import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.glm4_moe import (
    GlmMoeDsaForCausalLM,
    _modelopt_fp4_excludes_shared_experts,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeQuantConfig:
    def __init__(self, name, exclude_modules):
        self._name = name
        self.exclude_modules = exclude_modules

    def get_name(self):
        return self._name


class TestGlmSharedExpertFusionPolicy(unittest.TestCase):
    def test_modelopt_fp4_shared_expert_exclusion_disables_fusion(self):
        quant_config = _FakeQuantConfig(
            "modelopt_fp4",
            ["model.layers.*.mlp.shared_experts.*"],
        )

        self.assertTrue(_modelopt_fp4_excludes_shared_experts(quant_config))

    def test_modelopt_fp4_without_shared_expert_exclusion_keeps_fusion_available(self):
        quant_config = _FakeQuantConfig("modelopt_fp4", ["lm_head"])

        self.assertFalse(_modelopt_fp4_excludes_shared_experts(quant_config))

    def test_other_quantization_does_not_trigger_modelopt_fp4_guard(self):
        quant_config = _FakeQuantConfig(
            "modelopt_fp8",
            ["model.layers.*.mlp.shared_experts.*"],
        )

        self.assertFalse(_modelopt_fp4_excludes_shared_experts(quant_config))

    @patch("sglang.srt.models.glm4_moe.get_global_server_args")
    def test_glm_dsa_explicit_disable_shared_expert_fusion_sets_zero(self, mock_args):
        mock_args.return_value = SimpleNamespace(disable_shared_experts_fusion=True)
        model = object.__new__(GlmMoeDsaForCausalLM)
        model.quant_config = _FakeQuantConfig("modelopt_fp4", [])

        model.determine_num_fused_shared_experts()

        self.assertEqual(model.num_fused_shared_experts, 0)


if __name__ == "__main__":
    unittest.main()
