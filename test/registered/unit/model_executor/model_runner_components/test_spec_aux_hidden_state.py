import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _resolve_dflash_aux_hidden_state,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkAuxHiddenStateConfig(unittest.TestCase):
    def test_bundled_dspark_uses_target_layer_count_as_stage_count(self):
        server_args = SimpleNamespace(
            speculative_draft_model_path="/model",
            speculative_draft_model_revision=None,
        )
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=43)
        )
        draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=43,
                num_nextn_predict_layers=1,
                dspark_block_size=5,
                dspark_markov_rank=256,
                dspark_noise_token_id=128799,
                dspark_target_layer_ids=[40, 41, 42],
            ),
            num_nextn_predict_layers=1,
        )
        spec_algorithm = MagicMock()
        spec_algorithm.is_dflash_family.return_value = True
        spec_algorithm.is_dspark.return_value = True
        config = SpecAuxHiddenStateConfig()

        with (
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.ModelConfig.from_server_args",
                return_value=draft_model_config,
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state._resolve_dflash_draft_cell_size",
                return_value=123,
            ),
        ):
            _resolve_dflash_aux_hidden_state(
                config=config,
                server_args=server_args,
                model_config=model_config,
                spec_algorithm=spec_algorithm,
                is_draft_worker=False,
            )

        self.assertEqual(config.dflash_draft_num_layers, 3)
        self.assertEqual(config.dflash_target_layer_ids, [40, 41, 42])


if __name__ == "__main__":
    unittest.main()
