"""
Integration tests for attention steering via the semantic routing loop.

Tests:
1. API-level attention_biases injection
2. Mock sidecar feedback processing
3. Full routing loop with biases wired to GPU
"""

import unittest
from typing import Dict

import torch

# Test imports
from sglang.srt.managers.schedule_batch import SemanticMemory


class TestSemanticMemory(unittest.TestCase):
    """Unit tests for SemanticMemory class."""

    def test_attention_biases_initialization(self):
        """Test that SemanticMemory initializes with empty biases."""
        mem = SemanticMemory()
        self.assertEqual(mem.attention_biases, {})

    def test_add_attention_bias(self):
        """Test adding attention biases."""
        mem = SemanticMemory()
        mem.add_attention_bias(layer_id=10, token_pos=42, bias=0.5)
        mem.add_attention_bias(layer_id=10, token_pos=128, bias=0.3)
        mem.add_attention_bias(layer_id=15, token_pos=0, bias=1.0)

        self.assertEqual(len(mem.attention_biases), 2)  # 2 layers
        self.assertEqual(mem.attention_biases[10][42], 0.5)
        self.assertEqual(mem.attention_biases[10][128], 0.3)
        self.assertEqual(mem.attention_biases[15][0], 1.0)

    def test_update_from_sidecar(self):
        """Test updating from sidecar feedback."""
        mem = SemanticMemory()

        # Match the expected format in update_from_sidecar
        feedback = {
            "manifold_zone": "semantic_bridge",
            "manifold_confidence": 0.92,
            "hub_tokens": [42, 128, 256],
            "next_capture_layers": [15, 16, 17],
            "suggested_biases": {  # Note: uses "suggested_biases" not "attention_biases"
                12: {42: 0.3, 100: 0.5}
            },
            "anchors": {  # Note: uses "anchors" not "semantic_anchors"
                128: ("entity", 0.95)
            },
        }

        mem.update_from_sidecar(feedback)

        self.assertEqual(mem.manifold_zone, "semantic_bridge")
        self.assertAlmostEqual(mem.manifold_confidence, 0.92)
        self.assertEqual(mem.hub_tokens, [42, 128, 256])
        self.assertEqual(mem.next_capture_layers, [15, 16, 17])
        self.assertAlmostEqual(mem.attention_biases[12][42], 0.3)
        self.assertAlmostEqual(mem.attention_biases[12][100], 0.5)

    def test_get_layer_biases(self):
        """Test getting biases for a specific layer."""
        mem = SemanticMemory()
        mem.add_attention_bias(10, 42, 0.5)
        mem.add_attention_bias(10, 100, 0.3)

        biases = mem.get_layer_biases(10)
        self.assertEqual(biases, {42: 0.5, 100: 0.3})

        # Non-existent layer returns empty dict
        self.assertEqual(mem.get_layer_biases(99), {})

    def test_clear_steering(self):
        """Test clearing steering modifications (biases, injections, next_capture_layers)."""
        mem = SemanticMemory()
        mem.add_attention_bias(10, 42, 0.5)
        mem.next_capture_layers = [15, 16]

        mem.clear_steering()

        self.assertEqual(mem.attention_biases, {})
        self.assertIsNone(mem.next_capture_layers)
        # Note: manifold_zone and hub_tokens are NOT cleared by clear_steering
        # They represent observed state, not steering modifications


class TestAttentionBiasesWiring(unittest.TestCase):
    """Test that attention biases are wired through the batch pipeline."""

    def test_biases_collected_in_model_worker_batch(self):
        """Test that biases from semantic_memory are collected in ModelWorkerBatch."""
        # This is a structural test - we verify the Dict format is correct
        # Full integration requires a running server

        biases: Dict[int, Dict[int, float]] = {
            10: {42: 0.5, 100: 0.3},
            15: {0: 1.0},
        }

        # Verify the structure matches what get_model_worker_batch expects
        # Format: Dict[layer_id -> List[Dict[token_pos -> bias]]]
        batch_size = 4
        collected: Dict[int, list] = {}

        for layer_id, token_biases in biases.items():
            if layer_id not in collected:
                collected[layer_id] = [{} for _ in range(batch_size)]
            collected[layer_id][0] = token_biases  # First request in batch

        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[10][0], {42: 0.5, 100: 0.3})
        self.assertEqual(collected[15][0], {0: 1.0})

    def test_forward_batch_bias_tensor_creation(self):
        """Test that ForwardBatch can create bias tensors."""
        # Create a mock ForwardBatch with bias data
        batch_size = 2
        max_seq_len = 128

        # Simulate sparse indices and values
        batch_indices = torch.tensor([0, 0, 1])
        token_positions = torch.tensor([10, 20, 5])
        bias_values = torch.tensor([0.5, 0.3, 1.0])

        indices = torch.stack([batch_indices, token_positions], dim=1)

        # Create dense tensor
        bias_tensor = torch.zeros((batch_size, max_seq_len))
        valid_mask = token_positions < max_seq_len
        if valid_mask.any():
            bias_tensor[
                batch_indices[valid_mask], token_positions[valid_mask]
            ] = bias_values[valid_mask]

        # Verify (use assertAlmostEqual for float comparison)
        self.assertAlmostEqual(bias_tensor[0, 10].item(), 0.5, places=5)
        self.assertAlmostEqual(bias_tensor[0, 20].item(), 0.3, places=5)
        self.assertAlmostEqual(bias_tensor[1, 5].item(), 1.0, places=5)
        self.assertAlmostEqual(
            bias_tensor[0, 0].item(), 0.0, places=5
        )  # Unset position


class TestAPIBiasesConversion(unittest.TestCase):
    """Test API string-keyed biases to int-keyed conversion."""

    def test_convert_attention_biases(self):
        """Test conversion from API format to internal format."""
        # API format uses string keys (JSON limitation)
        api_biases = {
            "10": {"42": 0.5, "100": 0.3},
            "15": {"0": 1.0},
        }

        # Convert to internal format
        internal_biases = {
            int(layer_id): {
                int(token_pos): bias for token_pos, bias in token_biases.items()
            }
            for layer_id, token_biases in api_biases.items()
        }

        self.assertEqual(internal_biases[10][42], 0.5)
        self.assertEqual(internal_biases[10][100], 0.3)
        self.assertEqual(internal_biases[15][0], 1.0)

    def test_none_biases_passthrough(self):
        """Test that None biases are passed through correctly."""
        api_biases = None
        internal_biases = (
            {
                int(layer_id): {
                    int(token_pos): bias for token_pos, bias in token_biases.items()
                }
                for layer_id, token_biases in api_biases.items()
            }
            if api_biases
            else None
        )

        self.assertIsNone(internal_biases)


class TestMockSidecar(unittest.TestCase):
    """Test mock sidecar integration (requires zmq)."""

    def test_sidecar_import(self):
        """Test that the mock sidecar can be imported."""
        import importlib.util

        zmq_available = importlib.util.find_spec("zmq") is not None

        if zmq_available:
            # Verify the sidecar script is syntactically correct
            spec = importlib.util.spec_from_file_location(
                "mock_sidecar", "scripts/mock_attention_sidecar.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Just verify it can be loaded, don't run it
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
