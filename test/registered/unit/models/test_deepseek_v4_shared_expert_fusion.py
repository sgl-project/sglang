import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_context, get_exec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(unittest.TestCase):
    """The disable decision is a load-time resolution: it lands on the
    published config bag via declare_load_time_override (bag-only; the
    ServerArgs instance stays pristine)."""

    def _make_model(self, n_shared_experts=1):
        return SimpleNamespace(
            config=SimpleNamespace(n_shared_experts=n_shared_experts)
        )

    def _publish(self, enforce):
        override = get_context().override_server_args(
            enforce_shared_experts_fusion=enforce
        )
        override.install()
        self.addCleanup(override.restore)

    def test_disables_shared_fusion_without_enforce(self):
        self._publish(enforce=False)
        model = self._make_model()

        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 0)
        # post-init declaration lands on the published config bag
        self.assertTrue(get_exec().moe.disable_shared_experts_fusion)

    def test_enables_shared_fusion_when_enforced(self):
        self._publish(enforce=True)
        model = self._make_model()

        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 1)
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)


if __name__ == "__main__":
    unittest.main()
