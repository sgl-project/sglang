import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models import deepseek_v4 as deepseek_v4_module
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(unittest.TestCase):
    def _make_model(self, n_shared_experts=1):
        return SimpleNamespace(
            config=SimpleNamespace(n_shared_experts=n_shared_experts)
        )

    def test_disables_shared_fusion_without_enforce(self):
        server_args = SimpleNamespace(
            disable_shared_experts_fusion=False,
            enforce_shared_experts_fusion=False,
        )
        model = self._make_model()

        with patch.object(
            deepseek_v4_module, "get_global_server_args", return_value=server_args
        ):
            DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 0)
        self.assertTrue(server_args.disable_shared_experts_fusion)

    def test_enables_shared_fusion_when_enforced(self):
        server_args = SimpleNamespace(
            disable_shared_experts_fusion=False,
            enforce_shared_experts_fusion=True,
        )
        model = self._make_model()

        with patch.object(
            deepseek_v4_module, "get_global_server_args", return_value=server_args
        ):
            DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 1)
        self.assertFalse(server_args.disable_shared_experts_fusion)


if __name__ == "__main__":
    unittest.main()
