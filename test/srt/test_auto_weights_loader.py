"""
Unit tests for AutoWeightsLoader class.

This test module verifies the functionality of AutoWeightsLoader, which
simplifies weight loading by automatically detecting child modules and parameters.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from sglang.srt.models.utils import AutoWeightsLoader
from sglang.test.test_utils import CustomTestCase


class SimpleModule(nn.Module):
    """Simple test module with basic parameters."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 10))
        self.bias = nn.Parameter(torch.zeros(10))


class NestedModule(nn.Module):
    """Nested test module with child modules."""

    def __init__(self):
        super().__init__()
        self.layer1 = SimpleModule()
        self.layer2 = SimpleModule()
        self.final_weight = nn.Parameter(torch.zeros(5, 5))


class ModuleWithCustomLoader(nn.Module):
    """Module with custom load_weights method."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 10))
        self.loaded_weights = []

    def load_weights(self, weights):
        """Custom weight loading that tracks loaded weights."""
        for name, tensor in weights:
            self.loaded_weights.append(name)
        return self.loaded_weights


class TestAutoWeightsLoader(CustomTestCase):
    """Test cases for AutoWeightsLoader functionality."""

    def test_basic_weight_loading(self):
        """Test basic weight loading into a simple module."""
        module = SimpleModule()
        weights = [
            ("weight", torch.ones(10, 10)),
            ("bias", torch.ones(10)),
        ]

        loader = AutoWeightsLoader(module)
        loaded = loader.load_weights(iter(weights))

        # Check that weights were loaded
        self.assertEqual(len(loaded), 2)
        self.assertIn("weight", loaded)
        self.assertIn("bias", loaded)

        # Verify the actual values were loaded
        self.assertTrue(torch.allclose(module.weight, torch.ones(10, 10)))
        self.assertTrue(torch.allclose(module.bias, torch.ones(10)))

    def test_skip_prefixes(self):
        """Test that skip_prefixes correctly skips weights."""
        module = NestedModule()
        weights = [
            ("layer1.weight", torch.ones(10, 10)),
            ("layer1.bias", torch.ones(10)),
            ("layer2.weight", torch.ones(10, 10)),
            ("layer2.bias", torch.ones(10)),
            ("final_weight", torch.ones(5, 5)),
        ]

        # Skip layer1 entirely
        loader = AutoWeightsLoader(module, skip_prefixes=["layer1"])
        loaded = loader.load_weights(iter(weights))

        # Should only load layer2 and final_weight
        self.assertEqual(len(loaded), 3)
        self.assertNotIn("layer1.weight", loaded)
        self.assertNotIn("layer1.bias", loaded)
        self.assertIn("layer2.weight", loaded)
        self.assertIn("layer2.bias", loaded)
        self.assertIn("final_weight", loaded)

        # Verify layer1 wasn't loaded (still zeros)
        self.assertTrue(torch.allclose(module.layer1.weight, torch.zeros(10, 10)))
        # Verify layer2 was loaded
        self.assertTrue(torch.allclose(module.layer2.weight, torch.ones(10, 10)))

    def test_skip_substrs(self):
        """Test that skip_substrs correctly skips weights."""
        module = NestedModule()
        weights = [
            ("layer1.weight", torch.ones(10, 10)),
            ("layer1.bias", torch.ones(10)),
            ("layer2.weight", torch.ones(10, 10)),
            ("layer2.bias", torch.ones(10)),
        ]

        # Skip all bias parameters
        loader = AutoWeightsLoader(module, skip_substrs=["bias"])
        loaded = loader.load_weights(iter(weights))

        # Should only load weight parameters
        self.assertEqual(len(loaded), 2)
        self.assertIn("layer1.weight", loaded)
        self.assertIn("layer2.weight", loaded)
        self.assertNotIn("layer1.bias", loaded)
        self.assertNotIn("layer2.bias", loaded)

    def test_rotary_emb_auto_skip(self):
        """Test that rotary embedding weights are automatically skipped."""
        module = SimpleModule()
        weights = [
            ("weight", torch.ones(10, 10)),
            ("rotary_emb.inv_freq", torch.ones(5)),
            ("rotary_emb.cos_cached", torch.ones(5)),
            ("rotary_emb.sin_cached", torch.ones(5)),
        ]

        loader = AutoWeightsLoader(module)
        loaded = loader.load_weights(iter(weights))

        # Should only load the weight, skip rotary_emb weights
        self.assertEqual(len(loaded), 1)
        self.assertIn("weight", loaded)

    def test_nested_module_loading(self):
        """Test loading weights into nested modules."""
        module = NestedModule()
        weights = [
            ("layer1.weight", torch.ones(10, 10)),
            ("layer1.bias", torch.ones(10)),
            ("layer2.weight", torch.ones(10, 10) * 2),
            ("layer2.bias", torch.ones(10) * 2),
            ("final_weight", torch.ones(5, 5) * 3),
        ]

        loader = AutoWeightsLoader(module)
        loaded = loader.load_weights(iter(weights))

        # All weights should be loaded
        self.assertEqual(len(loaded), 5)

        # Verify correct values
        self.assertTrue(torch.allclose(module.layer1.weight, torch.ones(10, 10)))
        self.assertTrue(torch.allclose(module.layer2.weight, torch.ones(10, 10) * 2))
        self.assertTrue(torch.allclose(module.final_weight, torch.ones(5, 5) * 3))

    def test_custom_load_weights_method(self):
        """Test that modules with custom load_weights are handled correctly."""
        module = ModuleWithCustomLoader()
        weights = [
            ("weight", torch.ones(10, 10)),
        ]

        loader = AutoWeightsLoader(module)
        loaded = loader.load_weights(iter(weights))

        # The custom loader should have been called
        self.assertEqual(len(module.loaded_weights), 1)
        self.assertIn("weight", module.loaded_weights)

    def test_ignore_unexpected_prefixes(self):
        """Test that ignore_unexpected_prefixes correctly ignores missing weights."""
        module = SimpleModule()
        weights = [
            ("weight", torch.ones(10, 10)),
            ("vision_tower.weight", torch.ones(5, 5)),  # Not in module
            ("vision_tower.bias", torch.ones(5)),  # Not in module
        ]

        # Should not raise error for vision_tower weights
        loader = AutoWeightsLoader(module, ignore_unexpected_prefixes=["vision_tower"])
        loaded = loader.load_weights(iter(weights))

        # Should only load the actual module weight
        self.assertEqual(len(loaded), 1)
        self.assertIn("weight", loaded)

    def test_ignore_unexpected_suffixes(self):
        """Test that ignore_unexpected_suffixes correctly ignores missing weights."""
        module = SimpleModule()
        weights = [
            ("weight", torch.ones(10, 10)),
            ("extra.attn.bias", torch.ones(5)),  # Not in module
        ]

        # Should not raise error for .attn.bias weights
        loader = AutoWeightsLoader(module, ignore_unexpected_suffixes=[".attn.bias"])
        loaded = loader.load_weights(iter(weights))

        # Should only load the actual module weight
        self.assertEqual(len(loaded), 1)
        self.assertIn("weight", loaded)

    def test_error_on_unexpected_weight(self):
        """Test that an error is raised for unexpected weights without ignore rules."""
        module = SimpleModule()
        weights = [
            ("weight", torch.ones(10, 10)),
            ("unexpected_param", torch.ones(5, 5)),  # Not in module
        ]

        loader = AutoWeightsLoader(module)

        # Should raise ValueError for unexpected parameter
        with self.assertRaises(ValueError) as context:
            loader.load_weights(iter(weights))

        self.assertIn("unexpected_param", str(context.exception))

    def test_custom_weight_loader_attribute(self):
        """Test that custom weight_loader attributes are respected."""
        module = SimpleModule()

        # Add a custom weight loader to the parameter
        custom_loader_called = []

        def custom_loader(param, weight):
            custom_loader_called.append(True)
            param.data.copy_(weight * 2)  # Load with 2x multiplier

        module.weight.weight_loader = custom_loader

        weights = [
            ("weight", torch.ones(10, 10)),
        ]

        loader = AutoWeightsLoader(module)
        loader.load_weights(iter(weights))

        # Custom loader should have been called
        self.assertEqual(len(custom_loader_called), 1)
        # Weight should be 2x the input
        self.assertTrue(torch.allclose(module.weight, torch.ones(10, 10) * 2))

    def test_empty_weights(self):
        """Test handling of empty weight iterator."""
        module = SimpleModule()
        weights = []

        loader = AutoWeightsLoader(module)
        loaded = loader.load_weights(iter(weights))

        # Should handle empty iterator gracefully
        self.assertEqual(len(loaded), 0)

    def test_defensive_copy_of_lists(self):
        """Test that input lists are copied, not mutated."""
        module = SimpleModule()

        skip_prefixes = ["layer1"]
        skip_substrs = ["bias"]
        ignore_prefixes = ["vision"]
        ignore_suffixes = [".attn"]

        # Store original list IDs
        original_skip_prefixes_id = id(skip_prefixes)
        original_skip_substrs_id = id(skip_substrs)

        loader = AutoWeightsLoader(
            module,
            skip_prefixes=skip_prefixes,
            skip_substrs=skip_substrs,
            ignore_unexpected_prefixes=ignore_prefixes,
            ignore_unexpected_suffixes=ignore_suffixes,
        )

        # Original lists should not be modified
        self.assertEqual(len(skip_prefixes), 1)
        self.assertEqual(len(skip_substrs), 1)

        # Loader should have created new lists
        self.assertNotEqual(id(loader.skip_prefixes), original_skip_prefixes_id)
        self.assertNotEqual(id(loader.skip_substrs), original_skip_substrs_id)

        # But content should be preserved (plus auto-added rotary items)
        self.assertIn("layer1", loader.skip_prefixes)
        self.assertIn("bias", loader.skip_substrs)


if __name__ == "__main__":
    unittest.main()
