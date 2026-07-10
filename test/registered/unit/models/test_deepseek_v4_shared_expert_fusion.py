import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(unittest.TestCase):
    """The disable decision is a load-time resolution: it writes through to
    the published config via declare_load_time_override."""

    def setUp(self):
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def _make_model(self, n_shared_experts=1):
        return SimpleNamespace(
            config=SimpleNamespace(n_shared_experts=n_shared_experts)
        )

    def _publish(self, enforce):
        server_args = ServerArgs(model_path="dummy")
        server_args.enforce_shared_experts_fusion = enforce
        get_context().set_server_args(server_args)
        return server_args

    def test_disables_shared_fusion_without_enforce(self):
        server_args = self._publish(enforce=False)
        model = self._make_model()

        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 0)
        # post-init declaration writes through to the published config
        self.assertTrue(server_args.disable_shared_experts_fusion)

    def test_enables_shared_fusion_when_enforced(self):
        server_args = self._publish(enforce=True)
        model = self._make_model()

        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 1)
        self.assertFalse(server_args.disable_shared_experts_fusion)


if __name__ == "__main__":
    unittest.main()
