"""
Unit tests for fused parameter mapping in the model loader.

Tests the tracking_iterator logic that maps checkpoint weights with separate
q_proj/k_proj/v_proj to fused qkv_proj parameters in the model.
"""

import logging
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn


class FusedQKVModel(nn.Module):
    """Model with fused QKV projection (like optimized attention)."""

    def __init__(self):
        super().__init__()
        # Model has fused qkv_proj
        self.layer1 = Mock()
        self.layer1.qkv_proj = nn.Linear(768, 2304)  # 3 * 768 for Q, K, V
        self.layer2 = Mock()
        self.layer2.qkv_proj = nn.Linear(768, 2304)

    def load_weights(self, weights_iter):
        """Mock load_weights that consumes the iterator."""
        for name, tensor in weights_iter:
            pass  # Just consume the iterator


class FusedGateUpModel(nn.Module):
    """Model with fused gate_up projection (like MLP in LLaMA)."""

    def __init__(self):
        super().__init__()
        # Model has fused gate_up_proj
        self.mlp = Mock()
        self.mlp.gate_up_proj = nn.Linear(768, 2048)  # Combined gate and up


    def load_weights(self, weights_iter):
        """Mock load_weights that consumes the iterator."""
        for name, tensor in weights_iter:
            pass  # Just consume the iterator


class TestLoaderFusedParams(unittest.TestCase):
    """Tests for fused parameter mapping in loader's tracking_iterator."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.WARNING)

    @patch("sglang.srt.model_loader.loader.validate_loaded_weights")
    @patch("os.getenv")
    def test_qkv_fused_mapping(self, mock_getenv, mock_validate):
        """Test that separate q_proj/k_proj/v_proj weights map to fused qkv_proj.

        Models often fuse attention projections for efficiency. This verifies that
        checkpoint weights with separate Q, K, V projections are correctly recognized
        when the model uses a single qkv_proj parameter.
        """
        from sglang.srt.model_loader.loader import DefaultModelLoader

        mock_getenv.return_value = "0"  # Non-strict mode
        model = FusedQKVModel()

        # Simulate checkpoint with separate q_proj, k_proj, v_proj
        checkpoint_weights = [
            ("layer1.q_proj.weight", torch.randn(768, 768)),
            ("layer1.k_proj.weight", torch.randn(768, 768)),
            ("layer1.v_proj.weight", torch.randn(768, 768)),
            ("layer2.q_proj.weight", torch.randn(768, 768)),
            ("layer2.k_proj.weight", torch.randn(768, 768)),
            ("layer2.v_proj.weight", torch.randn(768, 768)),
        ]

        # Load weights (this will call tracking_iterator internally)
        DefaultModelLoader.load_weights_and_postprocess(
            model, iter(checkpoint_weights), "cpu"
        )

        # Verify that validate_loaded_weights was called
        self.assertTrue(mock_validate.called)

        # Get the loaded_param_names that were passed to validate_loaded_weights
        call_args = mock_validate.call_args
        loaded_param_names = call_args[0][1]  # Second argument

        # The fused parameters should be marked as loaded
        self.assertIn("layer1.qkv_proj.weight", loaded_param_names)
        self.assertIn("layer2.qkv_proj.weight", loaded_param_names)

        # Should have loaded both layers' fused params
        # Each layer has q/k/v mapping to the same qkv_proj, so we expect 2 entries
        qkv_params = [p for p in loaded_param_names if "qkv_proj" in p]
        self.assertEqual(len(qkv_params), 2)

    @patch("sglang.srt.model_loader.loader.validate_loaded_weights")
    @patch("os.getenv")
    def test_gate_up_fused_mapping(self, mock_getenv, mock_validate):
        """Test that gate_proj/up_proj in checkpoint map to gate_up_proj in model."""
        from sglang.srt.model_loader.loader import DefaultModelLoader

        mock_getenv.return_value = "0"  # Non-strict mode
        model = FusedGateUpModel()

        # Simulate checkpoint with separate gate_proj and up_proj
        checkpoint_weights = [
            ("mlp.gate_proj.weight", torch.randn(1024, 768)),
            ("mlp.up_proj.weight", torch.randn(1024, 768)),
        ]

        # Load weights
        DefaultModelLoader.load_weights_and_postprocess(
            model, iter(checkpoint_weights), "cpu"
        )

        # Get the loaded_param_names
        call_args = mock_validate.call_args
        loaded_param_names = call_args[0][1]

        # The fused parameter should be marked as loaded
        self.assertIn("mlp.gate_up_proj.weight", loaded_param_names)

    @patch("sglang.srt.model_loader.loader.validate_loaded_weights")
    @patch("os.getenv")
    def test_fused_mapping_with_many_layers(self, mock_getenv, mock_validate):
        """Test that fused parameter mapping scales efficiently with model size.

        Verifies that the mapping logic works correctly even with large models
        that have many layers with fused parameters.
        """
        from sglang.srt.model_loader.loader import DefaultModelLoader

        mock_getenv.return_value = "0"
        model = FusedQKVModel()

        # Simulate a large model with 100 layers
        for i in range(100):
            layer = Mock()
            layer.qkv_proj = nn.Linear(768, 2304)
            setattr(model, f"layer{i}", layer)

        # Create checkpoint with separate Q/K/V weights for each layer
        checkpoint_weights = []
        for i in range(100):
            checkpoint_weights.extend([
                (f"layer{i}.q_proj.weight", torch.randn(768, 768)),
                (f"layer{i}.k_proj.weight", torch.randn(768, 768)),
                (f"layer{i}.v_proj.weight", torch.randn(768, 768)),
            ])

        DefaultModelLoader.load_weights_and_postprocess(
            model, iter(checkpoint_weights), "cpu"
        )

        # Verify all fused params were correctly tracked
        call_args = mock_validate.call_args
        loaded_param_names = call_args[0][1]

        qkv_params = [p for p in loaded_param_names if "qkv_proj" in p]
        # 100 new layers + 2 original layers = 102 total
        self.assertEqual(len(qkv_params), 102)


if __name__ == "__main__":
    unittest.main()
