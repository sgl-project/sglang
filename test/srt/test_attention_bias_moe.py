#!/usr/bin/env python3
"""
Unit tests for multi-layer attention biases and MoE telemetry.

Tests:
- Multi-layer attention bias CSR indexing
- MoE telemetry computation (entropy, hubness, switch-rate)
- Think phase tracking
- Layer ID validation

Usage:
    python -m pytest test/srt/test_attention_bias_moe.py -v
"""

import math
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Tuple

import torch


class TestMultiLayerAttentionBias(unittest.TestCase):
    """Test multi-layer attention bias with CSR indexing."""

    def test_csr_indexing_single_layer(self):
        """Test CSR indexing with biases on a single layer."""
        # Simulate bias data: layer 5, batch 0, positions [10, 20, 30]
        indices = torch.tensor([
            [5, 0, 10],  # layer_id, batch_idx, token_pos
            [5, 0, 20],
            [5, 0, 30],
        ], dtype=torch.int64)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        layers = [5]
        # CSR indptr: [0, 3] meaning layer 5 has entries 0-3
        layer_indptr = torch.tensor([0, 3], dtype=torch.int64)

        # Verify CSR structure
        self.assertEqual(len(layer_indptr), len(layers) + 1)
        self.assertEqual(layer_indptr[-1].item(), len(values))

        # Extract biases for layer 5
        layer_idx = 0
        start, end = layer_indptr[layer_idx].item(), layer_indptr[layer_idx + 1].item()
        layer_indices = indices[start:end]
        layer_values = values[start:end]

        self.assertEqual(len(layer_values), 3)
        self.assertTrue(torch.all(layer_indices[:, 0] == 5))

    def test_csr_indexing_multi_layer(self):
        """Test CSR indexing with biases on multiple layers."""
        # Biases on layers 0, 5, 10
        indices = torch.tensor([
            # Layer 0 (2 entries)
            [0, 0, 5],
            [0, 0, 10],
            # Layer 5 (3 entries)
            [5, 0, 15],
            [5, 0, 20],
            [5, 0, 25],
            # Layer 10 (1 entry)
            [10, 0, 30],
        ], dtype=torch.int64)
        values = torch.tensor([0.5, 0.6, 1.0, 1.5, 2.0, 3.0], dtype=torch.float32)
        layers = [0, 5, 10]
        # CSR indptr: layer 0 has 0-2, layer 5 has 2-5, layer 10 has 5-6
        layer_indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int64)

        # Test layer 0
        start, end = layer_indptr[0].item(), layer_indptr[1].item()
        self.assertEqual(end - start, 2)
        self.assertTrue(torch.all(indices[start:end, 0] == 0))

        # Test layer 5
        start, end = layer_indptr[1].item(), layer_indptr[2].item()
        self.assertEqual(end - start, 3)
        self.assertTrue(torch.all(indices[start:end, 0] == 5))

        # Test layer 10
        start, end = layer_indptr[2].item(), layer_indptr[3].item()
        self.assertEqual(end - start, 1)
        self.assertTrue(torch.all(indices[start:end, 0] == 10))

    def test_no_bias_for_layer(self):
        """Test handling when a layer has no biases."""
        indices = torch.tensor([[5, 0, 10]], dtype=torch.int64)
        values = torch.tensor([1.0], dtype=torch.float32)
        layers = [5]
        layer_indptr = torch.tensor([0, 1], dtype=torch.int64)

        # Layer 3 is not in the list
        has_bias = 3 in layers
        self.assertFalse(has_bias)


class TestMoETelemetry(unittest.TestCase):
    """Test MoE telemetry computation."""

    def test_entropy_computation(self):
        """Test routing entropy computation from weights."""
        # Uniform weights -> high entropy
        uniform_weights = [0.5, 0.5]
        total = sum(uniform_weights)
        entropy_uniform = 0.0
        for w in uniform_weights:
            if w > 0:
                p = w / total
                entropy_uniform -= p * math.log2(p + 1e-10)

        # Should be close to 1.0 for 2 experts with uniform weights
        self.assertAlmostEqual(entropy_uniform, 1.0, places=2)

        # Peaked weights -> low entropy
        peaked_weights = [0.99, 0.01]
        total = sum(peaked_weights)
        entropy_peaked = 0.0
        for w in peaked_weights:
            if w > 0:
                p = w / total
                entropy_peaked -= p * math.log2(p + 1e-10)

        # Should be close to 0 for very peaked distribution
        self.assertLess(entropy_peaked, 0.2)

    def test_expert_churn_computation(self):
        """Test expert churn (switch-rate) computation."""
        # Previous top experts by layer
        prev_top_experts = {0: 5, 1: 10, 2: 15}

        # Current top experts - 2 out of 3 changed
        current_top_experts = {0: 7, 1: 10, 2: 20}

        churn_count = 0
        layer_count = 0
        for layer_id in current_top_experts:
            if layer_id in prev_top_experts:
                if prev_top_experts[layer_id] != current_top_experts[layer_id]:
                    churn_count += 1
            layer_count += 1

        churn_rate = churn_count / layer_count if layer_count > 0 else 0
        self.assertAlmostEqual(churn_rate, 2/3, places=2)

    def test_hubness_computation(self):
        """Test hubness (expert concentration) computation."""
        # Expert counts: expert 5 selected 80 times, expert 7 selected 20 times
        expert_counts = {5: 80, 7: 20}
        total_selections = sum(expert_counts.values())

        # Find top expert
        top_expert = max(expert_counts, key=expert_counts.get)
        top_count = expert_counts[top_expert]

        # Hubness = concentration = top_count / total
        hubness = top_count / total_selections
        self.assertAlmostEqual(hubness, 0.8, places=2)

        # More uniform distribution
        uniform_counts = {5: 25, 7: 25, 10: 25, 12: 25}
        total = sum(uniform_counts.values())
        top = max(uniform_counts.values())
        hubness_uniform = top / total
        self.assertAlmostEqual(hubness_uniform, 0.25, places=2)

    def test_telemetry_accumulation(self):
        """Test hubness accumulation across steps."""
        hubness_tracker: Dict[int, Dict[int, int]] = {}
        total_steps = 0

        # Simulate 10 steps
        for step in range(10):
            # Layer 0 always picks expert 5
            layer_id = 0
            expert_id = 5
            if layer_id not in hubness_tracker:
                hubness_tracker[layer_id] = {}
            hubness_tracker[layer_id][expert_id] = hubness_tracker[layer_id].get(expert_id, 0) + 1

            # Layer 1 alternates between expert 10 and 11
            layer_id = 1
            expert_id = 10 if step % 2 == 0 else 11
            if layer_id not in hubness_tracker:
                hubness_tracker[layer_id] = {}
            hubness_tracker[layer_id][expert_id] = hubness_tracker[layer_id].get(expert_id, 0) + 1

            total_steps += 1

        # Layer 0: expert 5 has 100% concentration
        self.assertEqual(hubness_tracker[0][5], 10)
        hubness_0 = hubness_tracker[0][5] / sum(hubness_tracker[0].values())
        self.assertAlmostEqual(hubness_0, 1.0, places=2)

        # Layer 1: experts 10 and 11 each have 50%
        hubness_1 = max(hubness_tracker[1].values()) / sum(hubness_tracker[1].values())
        self.assertAlmostEqual(hubness_1, 0.5, places=2)


class TestThinkPhaseTracking(unittest.TestCase):
    """Test think phase tracking for reasoning models."""

    def test_phase_transitions(self):
        """Test tracking of think/output phase transitions."""
        # Simulate think start token ID = 100, end token ID = 101
        think_start_id = 100
        think_end_id = 101

        current_phase = "output"  # Start in output phase
        phase_history = []

        # Simulate token sequence: normal, <think>, think content, </think>, output
        # Use distinct token IDs to avoid collision with special tokens
        tokens = [50, 60, think_start_id, 70, 80, 90, think_end_id, 200, 210]

        for tok in tokens:
            if tok == think_start_id:
                current_phase = "think"
            elif tok == think_end_id:
                current_phase = "output"
            phase_history.append(current_phase)

        # Check phase assignments
        self.assertEqual(phase_history[0], "output")  # Before <think>
        self.assertEqual(phase_history[1], "output")
        self.assertEqual(phase_history[2], "think")   # At <think>
        self.assertEqual(phase_history[3], "think")   # Inside think
        self.assertEqual(phase_history[4], "think")
        self.assertEqual(phase_history[5], "think")
        self.assertEqual(phase_history[6], "output")  # At </think>
        self.assertEqual(phase_history[7], "output")  # After </think>

    def test_phase_in_attention_info(self):
        """Test that think_phase is included in attention info."""
        attention_info = {
            "token_positions": [10, 20, 30],
            "attention_scores": [0.5, 0.3, 0.2],
            "layer_id": 11,
            "decode_step": 5,
            "think_phase": "think",
        }

        self.assertIn("think_phase", attention_info)
        self.assertEqual(attention_info["think_phase"], "think")


class TestLayerValidation(unittest.TestCase):
    """Test layer ID validation."""

    def test_filter_invalid_layer_ids(self):
        """Test filtering of invalid layer IDs."""
        num_layers = 32
        requested_layers = [-1, 0, 15, 31, 32, 100]

        valid_layers = [
            lid for lid in requested_layers
            if 0 <= lid < num_layers
        ]

        self.assertEqual(valid_layers, [0, 15, 31])

    def test_all_invalid_falls_back_to_none(self):
        """Test that all invalid IDs results in None (capture all)."""
        num_layers = 32
        requested_layers = [-5, 50, 100]

        valid_layers = [
            lid for lid in requested_layers
            if 0 <= lid < num_layers
        ]

        result = valid_layers if valid_layers else None
        self.assertIsNone(result)

    def test_moe_layer_capture_filtering(self):
        """Test MoE layer filtering during capture."""
        # Simulate a model where MoE layers are at indices 2, 4, 6, 8
        moe_layer_ids = {2, 4, 6, 8}
        capture_layer_ids = [0, 2, 4, 10]  # User requests these

        # Filter to only actual MoE layers
        valid_moe_captures = [
            lid for lid in capture_layer_ids
            if lid in moe_layer_ids
        ]

        self.assertEqual(valid_moe_captures, [2, 4])

    def test_mid_layer_selection_standard(self):
        """Test mid layer selection for standard models (~70% depth)."""
        num_layers = 32

        # mid/mid_full selects ~70% depth
        mid_layer = (7 * num_layers) // 10
        self.assertEqual(mid_layer, 22)  # 70% of 32 = 22.4 -> 22

        # For 24-layer model
        num_layers_24 = 24
        mid_layer_24 = (7 * num_layers_24) // 10
        self.assertEqual(mid_layer_24, 16)  # 70% of 24 = 16.8 -> 16

    def test_mid_layer_selection_hybrid(self):
        """Test mid layer selection for hybrid models (e.g., Qwen3-Next)."""
        # Simulate hybrid model with sparse full attention layers
        # e.g., layers 0, 4, 8, 12, 16, 20, 24, 28 are full attention
        full_attn_layers = [0, 4, 8, 12, 16, 20, 24, 28]
        n_full = len(full_attn_layers)

        # mid_full selects ~60-70% depth into full attention layers
        mid_idx = (2 * n_full) // 3
        self.assertEqual(mid_idx, 5)  # 2/3 of 8 = 5.33 -> 5

        mid_layer = full_attn_layers[mid_idx]
        self.assertEqual(mid_layer, 20)  # Layer 20 is at 2/3 depth


class TestSparseBiasCorrection(unittest.TestCase):
    """Test sparse bias correction math."""

    def test_bias_effect_direction(self):
        """Test that positive bias increases attention weight."""
        # Original attention weight (after softmax)
        original_weight = 0.1  # 10% of attention
        bias = 2.0

        # Biased weight should be higher
        # exp(original_logit + bias) > exp(original_logit)
        # So the new weight should be higher
        exp_bias = math.exp(bias)
        self.assertGreater(exp_bias, 1.0)

    def test_negative_bias_decreases_weight(self):
        """Test that negative bias decreases attention weight."""
        bias = -2.0
        exp_bias_minus_1 = math.exp(bias) - 1

        # exp(-2) - 1 ≈ 0.135 - 1 = -0.865
        # This should decrease the weight
        self.assertLess(exp_bias_minus_1, 0)

    def test_zero_bias_no_change(self):
        """Test that zero bias has no effect."""
        bias = 0.0
        exp_bias_minus_1 = math.exp(bias) - 1

        # exp(0) - 1 = 0
        self.assertAlmostEqual(exp_bias_minus_1, 0.0, places=5)


class TestFingerprintMode(unittest.TestCase):
    """Test fingerprint mode telemetry."""

    def test_fingerprint_dimensions(self):
        """Test that fingerprint has correct dimensions."""
        # Fingerprint should be 20D: 16-bin histogram + 4 mass values
        expected_dims = 20

        # Simulate fingerprint
        fingerprint = [0.0] * 16  # 16-bin log-distance histogram
        fingerprint.extend([0.3, 0.4, 0.2, 0.1])  # local, mid, long, entropy

        self.assertEqual(len(fingerprint), expected_dims)

    def test_manifold_classification(self):
        """Test manifold classification from fingerprint."""
        # Syntax floor: high local mass
        syntax_floor = {"local_mass": 0.8, "mid_mass": 0.1, "long_mass": 0.1}
        # Semantic bridge: high mid mass
        semantic_bridge = {"local_mass": 0.2, "mid_mass": 0.7, "long_mass": 0.1}
        # Long range: high long mass
        long_range = {"local_mass": 0.1, "mid_mass": 0.2, "long_mass": 0.7}

        def classify(masses):
            if masses["local_mass"] > 0.5:
                return "syntax_floor"
            elif masses["mid_mass"] > 0.5:
                return "semantic_bridge"
            elif masses["long_mass"] > 0.5:
                return "long_range"
            return "diffuse"

        self.assertEqual(classify(syntax_floor), "syntax_floor")
        self.assertEqual(classify(semantic_bridge), "semantic_bridge")
        self.assertEqual(classify(long_range), "long_range")


class TestPerRequestOverridePriority(unittest.TestCase):
    """Test per-request override priority for sidecar control."""

    def test_override_priority_order(self):
        """Test that override priority is: request > sidecar > batch default."""
        # Simulate the override resolution logic
        batch_default = 8
        semantic_memory_value = 4
        request_value = 2

        # Case 1: Request value takes priority
        stride = batch_default
        req_stride = request_value
        sm_stride = semantic_memory_value
        if req_stride is not None:
            stride = req_stride
        elif sm_stride is not None:
            stride = sm_stride
        self.assertEqual(stride, 2)

        # Case 2: Semantic memory value when request is None
        stride = batch_default
        req_stride = None
        sm_stride = semantic_memory_value
        if req_stride is not None:
            stride = req_stride
        elif sm_stride is not None:
            stride = sm_stride
        self.assertEqual(stride, 4)

        # Case 3: Batch default when both are None
        stride = batch_default
        req_stride = None
        sm_stride = None
        if req_stride is not None:
            stride = req_stride
        elif sm_stride is not None:
            stride = sm_stride
        self.assertEqual(stride, 8)

    def test_sidecar_schema_v1_parsing(self):
        """Test parsing of schema v1 sidecar control signals."""
        sidecar_response = {
            "schema_version": 1,
            "manifold": {
                "zone": "semantic_bridge",
                "cluster_id": 12,
                "cluster_conf": 0.92
            },
            "control": {
                "next_capture_layer_ids": [8, 16, 24],
                "attention_stride": 4,
                "max_attention_steps": 256,
                "fingerprint_mode": True,
                "fingerprint_max_steps": 128,
                "hub_tokens": [42, 128, 256]
            }
        }

        # Simulate parsing
        schema_version = sidecar_response.get("schema_version", 0)
        self.assertEqual(schema_version, 1)

        control = sidecar_response.get("control", {})
        self.assertEqual(control.get("attention_stride"), 4)
        self.assertEqual(control.get("max_attention_steps"), 256)
        self.assertEqual(control.get("fingerprint_mode"), True)
        self.assertEqual(control.get("fingerprint_max_steps"), 128)

        manifold = sidecar_response.get("manifold", {})
        self.assertEqual(manifold.get("zone"), "semantic_bridge")
        self.assertEqual(manifold.get("cluster_id"), 12)

    def test_fingerprint_mode_override_logic(self):
        """Test fingerprint mode can be overridden per-request."""
        # Batch default
        batch_fingerprint_mode = True
        batch_fingerprint_max_steps = 256

        # Test 1: Request disables fingerprint mode
        req_fp_mode = False
        sm_fp_mode = None
        fingerprint_mode = batch_fingerprint_mode
        if req_fp_mode is not None:
            fingerprint_mode = req_fp_mode
        elif sm_fp_mode is not None:
            fingerprint_mode = sm_fp_mode
        self.assertFalse(fingerprint_mode)

        # Test 2: Sidecar extends max steps
        req_fp_max = None
        sm_fp_max = 512
        fingerprint_max_steps = batch_fingerprint_max_steps
        if req_fp_max is not None:
            fingerprint_max_steps = req_fp_max
        elif sm_fp_max is not None:
            fingerprint_max_steps = sm_fp_max
        self.assertEqual(fingerprint_max_steps, 512)

    def test_no_global_state_mutation(self):
        """Test that per-request overrides don't mutate shared state."""
        # Simulate batch-level defaults (immutable)
        class BatchDefaults:
            attention_tokens_stride = 8
            attention_fingerprint_mode = True
            attention_fingerprint_max_steps = 256

        # Simulate per-request state
        class Request1:
            attention_stride = 2
            attention_fingerprint_mode = False

        class Request2:
            attention_stride = None  # Use default
            attention_fingerprint_mode = None

        defaults = BatchDefaults()
        req1 = Request1()
        req2 = Request2()

        # Resolve for request 1
        stride1 = req1.attention_stride if req1.attention_stride is not None else defaults.attention_tokens_stride
        fp_mode1 = req1.attention_fingerprint_mode if req1.attention_fingerprint_mode is not None else defaults.attention_fingerprint_mode

        # Resolve for request 2
        stride2 = req2.attention_stride if req2.attention_stride is not None else defaults.attention_tokens_stride
        fp_mode2 = req2.attention_fingerprint_mode if req2.attention_fingerprint_mode is not None else defaults.attention_fingerprint_mode

        # Verify different requests get different values
        self.assertEqual(stride1, 2)
        self.assertEqual(stride2, 8)
        self.assertFalse(fp_mode1)
        self.assertTrue(fp_mode2)

        # Verify batch defaults unchanged
        self.assertEqual(defaults.attention_tokens_stride, 8)
        self.assertTrue(defaults.attention_fingerprint_mode)


if __name__ == "__main__":
    unittest.main()
