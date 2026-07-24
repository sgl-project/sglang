import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.srt.models.llama_eagle3 import LlamaModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLlamaEagle3NormBeforeFC(unittest.TestCase):
    def test_creates_input_norm_for_concatenated_aux_hidden_states(self):
        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=0,
            rms_norm_eps=1e-5,
            eagle_config={
                "eagle_aux_hidden_state_layer_ids": [24, 30, 36],
                "norm_before_fc": True,
            },
        )

        with patch(
            "sglang.srt.models.llama_eagle3.VocabParallelEmbedding",
            side_effect=lambda vocab_size, hidden_size, prefix: nn.Embedding(
                vocab_size, hidden_size
            ),
        ):
            model = LlamaModel(config)

        self.assertTrue(model.norm_before_fc)
        self.assertIsNotNone(model.input_norm)
        self.assertEqual(model.input_norm.weight.numel(), 3 * config.hidden_size)
        self.assertEqual(model.fc.in_features, 3 * config.hidden_size)


if __name__ == "__main__":
    unittest.main()
