import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGPTQConfig:
    def get_name(self):
        return "gptq"


class TestQwen3_5UnknownBiasWarning(unittest.TestCase):
    def test_unquantized_unknown_bias_logs_warning(self):
        """When not using GPTQ, unknown .bias should log a warning instead of being silently ignored."""
        model = MagicMock(spec=Qwen3_5ForConditionalGeneration)
        model.quant_config = None
        model.config = MagicMock()
        model.config.tie_word_embeddings = False
        model.named_parameters.return_value = []
        model.pp_group = MagicMock()
        model.pp_group.is_last_rank = False
        model.pooler = None

        dummy_weight = torch.zeros(4)
        weights = [
            ("model.visual.merger.linear_fc2_renamed.bias", dummy_weight),
            ("model.layers.0.self_attn.unknown.bias", dummy_weight),
        ]

        with patch("sglang.srt.models.qwen3_5.logger.warning") as mock_warning:
            Qwen3_5ForConditionalGeneration.load_weights(model, weights)
            logged_messages = [call.args[0] for call in mock_warning.call_args_list]

        self.assertTrue(
            any(
                "visual.merger.linear_fc2_renamed.bias" in msg
                for msg in logged_messages
            ),
            f"Expected warning for visual .bias, got: {logged_messages}",
        )
        self.assertTrue(
            any(
                "model.layers.0.self_attn.unknown.bias" in msg
                for msg in logged_messages
            ),
            f"Expected warning for text .bias, got: {logged_messages}",
        )

    def test_gptq_skips_text_bias_but_warns_on_visual_bias(self):
        """Under GPTQ, extra text biases are skipped silently, but visual biases should still warn."""
        model = MagicMock(spec=Qwen3_5ForConditionalGeneration)
        model.quant_config = _FakeGPTQConfig()
        model.config = MagicMock()
        model.config.tie_word_embeddings = False
        model.named_parameters.return_value = []
        model.pp_group = MagicMock()
        model.pp_group.is_last_rank = False
        model.pooler = None

        dummy_weight = torch.zeros(4)
        weights = [
            ("model.layers.0.self_attn.q_proj.bias", dummy_weight),
            ("model.visual.merger.linear_fc2_renamed.bias", dummy_weight),
        ]

        with patch("sglang.srt.models.qwen3_5.logger.warning") as mock_warning:
            Qwen3_5ForConditionalGeneration.load_weights(model, weights)
            logged_messages = [call.args[0] for call in mock_warning.call_args_list]

        # Text bias should be silently skipped under GPTQ
        self.assertFalse(
            any("q_proj.bias" in msg for msg in logged_messages),
            f"GPTQ extra text bias should not warn, but got: {logged_messages}",
        )
        # Visual bias should still warn under GPTQ
        self.assertTrue(
            any(
                "visual.merger.linear_fc2_renamed.bias" in msg
                for msg in logged_messages
            ),
            f"Expected warning for visual .bias under GPTQ, got: {logged_messages}",
        )


if __name__ == "__main__":
    unittest.main()
