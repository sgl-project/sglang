import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    resolve_spec_aux_hidden_state_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpecAuxHiddenState(unittest.TestCase):
    def setUp(self):
        self.server_args = SimpleNamespace(
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
        )
        self.model_config = SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(eagle_config=None),
        )
        self.spec_algorithm = MagicMock()
        self.spec_algorithm.is_eagle.return_value = True
        self.spec_algorithm.is_standalone.return_value = False
        self.spec_algorithm.is_eagle3.return_value = False
        self.spec_algorithm.is_dflash_family.return_value = False

    def resolve(self, *, is_draft_worker=False):
        return resolve_spec_aux_hidden_state_config(
            server_args=self.server_args,
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=is_draft_worker,
        )

    @patch(
        "sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state.ModelConfig.from_server_args"
    )
    def test_native_mtp_uses_target_nextn_layer_count(self, from_server_args):
        config = self.resolve()

        self.assertEqual(config.eagle_draft_num_layers, 1)
        from_server_args.assert_not_called()

    def test_draft_worker_does_not_budget_itself(self):
        config = self.resolve(is_draft_worker=True)

        self.assertIsNone(config.eagle_draft_num_layers)


if __name__ == "__main__":
    unittest.main()
