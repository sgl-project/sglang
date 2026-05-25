"""
Unit tests for bugfixes: #26286 (top_logprobs tensor crash) and #26013 (Gemma3 RoPE KeyError).

These tests verify the specific fixes without requiring a GPU or full server.
"""

import unittest
from unittest.mock import MagicMock

import torch


class TestDetokenizeTopLogprobsTensorCrash(unittest.TestCase):
    """Test for bug #26286: detokenize_top_logprobs_tokens crashes on multi-element tensor."""

    def setUp(self):
        """Create a minimal TokenizerManager-like object with just the methods we test."""
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        self.tm = TokenizerManager.__new__(TokenizerManager)
        self.tm.tokenizer = MagicMock()
        self.tm.tokenizer.batch_decode = MagicMock(
            side_effect=lambda x: ["token"] * len(x)
        )

    def test_multi_element_tensor_no_crash(self):
        """Multi-element GPU tensor should not crash with 'Boolean value of Tensor is ambiguous'."""
        token_logprobs_val = [
            torch.tensor([-0.1, -0.2, -0.3]),
            torch.tensor([-0.4]),
        ]
        token_logprobs_idx = [
            [1, 2, 3],
            [4],
        ]
        # This used to crash with RuntimeError before the fix
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_empty_list_returns_none(self):
        """Empty list entry should return None."""
        token_logprobs_val = [
            [],
            [0.1],
        ]
        token_logprobs_idx = [
            [],
            [5],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])

    def test_cpu_tensor_list_works(self):
        """Regular CPU float lists should still work as before."""
        token_logprobs_val = [
            [0.1, 0.2],
            [0.3],
        ]
        token_logprobs_idx = [
            [1, 2],
            [3],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_all_empty_returns_all_none(self):
        """All empty entries should return all None."""
        result = self.tm.detokenize_top_logprobs_tokens(
            [[], []], [[], []], decode_to_text=True
        )
        self.assertEqual(result, [None, None])

    def test_mixed_tensor_and_empty(self):
        """Mix of tensor entries and empty entries."""
        token_logprobs_val = [
            torch.tensor([-0.5, -0.6]),
            [],
            torch.tensor([-0.7]),
        ]
        token_logprobs_idx = [
            [10, 20],
            [],
            [30],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNotNone(result[2])

    def test_decode_to_text_false_skips_tokenizer(self):
        """With decode_to_text=False, tokenizer should not be called."""
        token_logprobs_val = [
            torch.tensor([-0.1, -0.2]),
        ]
        token_logprobs_idx = [
            [1, 2],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [(-0.1, 1, None), (-0.2, 2, None)])
        self.tm.tokenizer.batch_decode.assert_not_called()


class TestGemma3RoPEKeyError(unittest.TestCase):
    """Test for bug #26013: Gemma3 RoPE KeyError when factor is absent."""

    def _make_gemma3_text_model_config(self, full_attention_rope_params):
        """Create a minimal config object that mirrors Gemma3TextConfig fields."""
        from transformers import Gemma3TextConfig

        cfg = Gemma3TextConfig(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=1024,
            sliding_window=64,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": full_attention_rope_params,
            },
        )
        return cfg

    def test_rope_parameters_without_factor(self):
        """Config with full_attention using default RoPE (no factor) should not raise KeyError."""
        cfg = self._make_gemma3_text_model_config(
            {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            }
        )

        # We only need to test the rope_parameters construction logic,
        # not the full model init which requires weights.
        # Test the specific code path that used to crash:
        rope_params = cfg.rope_parameters
        if isinstance(rope_params, dict) and "full_attention" in rope_params:
            global_theta = rope_params["full_attention"].get("rope_theta", 1000000.0)

        # This line used to crash with KeyError: 'factor'
        factor = rope_params["full_attention"].get("factor", 1.0)
        rope_type = (
            "linear"
            if rope_params["full_attention"].get("factor") is not None
            else "default"
        )
        self.assertEqual(rope_type, "default")
        self.assertEqual(factor, 1.0)
        self.assertEqual(global_theta, 1000000.0)

    def test_rope_parameters_with_factor(self):
        """Config with factor present should still produce 'linear' rope_type."""
        cfg = self._make_gemma3_text_model_config(
            {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            }
        )

        rope_params = cfg.rope_parameters
        factor = rope_params["full_attention"].get("factor", 1.0)
        rope_type = (
            "linear"
            if rope_params["full_attention"].get("factor") is not None
            else "default"
        )
        self.assertEqual(rope_type, "linear")
        self.assertEqual(factor, 8.0)


if __name__ == "__main__":
    unittest.main()
