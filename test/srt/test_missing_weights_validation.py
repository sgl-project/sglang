"""
Unit tests for missing weights validation logic.

Tests the validation logic that detects when model parameters are not loaded
from checkpoints, which would leave them with random initialization values.
"""

import logging
import unittest
from unittest.mock import MagicMock, Mock

import torch
import torch.nn as nn

from sglang.srt.model_loader.weight_utils import validate_loaded_weights


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, tie_weights=False):
        super().__init__()
        self.config = Mock()
        self.config.tie_word_embeddings = tie_weights

        # Regular parameters that should be loaded
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)

        # Tied weights scenario
        if tie_weights:
            self.embed_tokens = nn.Embedding(100, 10)
            self.lm_head = nn.Linear(10, 100, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

        # Computed parameter (like rotary embeddings)
        self.register_buffer("rotary_emb.inv_freq", torch.randn(5))


class TestMissingWeightsValidation(unittest.TestCase):
    """Tests for validate_loaded_weights function."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    def test_all_weights_loaded(self):
        """Test that validation passes when all required weights are loaded."""
        model = DummyModel()

        # Simulate all weights being loaded
        loaded_params = {
            "layer1.weight",
            "layer1.bias",
            "layer2.weight",
            "layer2.bias",
        }

        # Should not raise in strict mode
        validate_loaded_weights(model, loaded_params, self.logger, strict=True)

    def test_missing_weights_non_strict(self):
        """Test that missing weights issue a warning in non-strict mode."""
        model = DummyModel()

        # Simulate only some weights being loaded (layer2 missing)
        loaded_params = {
            "layer1.weight",
            "layer1.bias",
        }

        # Should only warn, not raise
        with self.assertLogs(self.logger, level="WARNING") as cm:
            validate_loaded_weights(model, loaded_params, self.logger, strict=False)

        # Check warning message contains info about missing params
        self.assertTrue(any("layer2.weight" in msg for msg in cm.output))
        self.assertTrue(any("layer2.bias" in msg for msg in cm.output))

    def test_missing_weights_strict_mode(self):
        """Test that missing weights raise an error in strict mode."""
        model = DummyModel()

        # Simulate only some weights being loaded
        loaded_params = {
            "layer1.weight",
            "layer1.bias",
        }

        # Should raise ValueError in strict mode
        with self.assertRaises(ValueError) as cm:
            validate_loaded_weights(model, loaded_params, self.logger, strict=True)

        error_msg = str(cm.exception)
        self.assertIn("layer2.weight", error_msg)
        self.assertIn("layer2.bias", error_msg)
        self.assertIn("random values", error_msg.lower())

    def test_tied_weights_excluded(self):
        """Test that tied weights (lm_head) are excluded from validation."""
        model = DummyModel(tie_weights=True)

        # Only load embed_tokens, not lm_head (which is tied)
        loaded_params = {
            "layer1.weight",
            "layer1.bias",
            "layer2.weight",
            "layer2.bias",
            "embed_tokens.weight",
            # Note: lm_head.weight is NOT loaded (it's tied to embed_tokens.weight)
        }

        # Should not raise - lm_head.weight is expected to be missing
        validate_loaded_weights(model, loaded_params, self.logger, strict=True)

    def test_computed_params_excluded(self):
        """Test that computed parameters (rotary_emb) are excluded."""
        model = DummyModel()

        # Don't include rotary_emb.inv_freq in loaded params
        loaded_params = {
            "layer1.weight",
            "layer1.bias",
            "layer2.weight",
            "layer2.bias",
            # Note: rotary_emb.inv_freq is NOT in loaded_params (it's computed)
        }

        # Should not raise - computed params are expected to be missing
        validate_loaded_weights(model, loaded_params, self.logger, strict=True)

    def test_partial_loading_detected(self):
        """Test that partially loaded models are detected."""
        model = DummyModel()

        # Simulate a bug where layer2 weights weren't in checkpoint
        loaded_params = {
            "layer1.weight",
            "layer1.bias",
            # layer2 completely missing - this is the bug we're catching!
        }

        # Should raise in strict mode
        with self.assertRaises(ValueError) as cm:
            validate_loaded_weights(model, loaded_params, self.logger, strict=True)

        error_msg = str(cm.exception)
        # Should mention both missing parameters
        self.assertIn("layer2.weight", error_msg)
        self.assertIn("layer2.bias", error_msg)
        # Should explain the issue
        self.assertIn("not loaded from checkpoint", error_msg.lower())


if __name__ == "__main__":
    unittest.main()
