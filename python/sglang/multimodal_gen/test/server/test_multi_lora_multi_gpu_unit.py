# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for multi-LoRA multi-GPU functionality.

These tests verify the synchronization and broadcast logic for multi-LoRA
in tensor parallel (TP) setups without requiring actual GPU hardware.
They mock the distributed primitives to test the logic correctness.

Run with:
    pytest python/sglang/multimodal_gen/test/server/test_multi_lora_multi_gpu_unit.py -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import torch


@dataclass
class MockReq:
    """Mock request object for testing."""

    request_id: str
    lora_nickname: Optional[str] = None
    lora_path: Optional[str] = None


class MockTPGroup:
    """Mock tensor parallel group for testing broadcast logic."""

    def __init__(self, rank: int = 0, world_size: int = 2):
        self.rank_in_group = rank
        self.world_size = world_size
        self._broadcast_calls = []
        self._tensor_broadcast_calls = []

    def broadcast_object(self, obj, src: int = 0):
        """Mock broadcast_object that simulates multi-rank behavior."""
        self._broadcast_calls.append({"obj": obj, "src": src})
        # In real distributed setting, all ranks would receive the same object
        # For testing, we just return the object (simulating rank 0 behavior)
        return obj

    def broadcast_tensor_dict(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], src: int = 0
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Mock broadcast_tensor_dict."""
        self._tensor_broadcast_calls.append({"tensor_dict": tensor_dict, "src": src})
        return tensor_dict


class TestDiffusionLoRAManagerMultiGPU(unittest.TestCase):
    """Test DiffusionLoRAManager's multi-GPU synchronization."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch distributed primitives to avoid actual distributed initialization
        self.dist_patcher = patch(
            "sglang.multimodal_gen.runtime.lora.lora_manager.dist"
        )
        self.mock_dist = self.dist_patcher.start()
        self.mock_dist.is_initialized.return_value = True
        self.mock_dist.get_rank.return_value = 0

        self.tp_group_patcher = patch(
            "sglang.multimodal_gen.runtime.lora.lora_manager.get_tp_group"
        )
        self.mock_get_tp_group = self.tp_group_patcher.start()

        self.tp_rank_patcher = patch(
            "sglang.multimodal_gen.runtime.lora.lora_manager.get_tp_rank"
        )
        self.mock_get_tp_rank = self.tp_rank_patcher.start()
        self.mock_get_tp_rank.return_value = 0

        self.tp_size_patcher = patch(
            "sglang.multimodal_gen.runtime.lora.lora_manager.get_tp_world_size"
        )
        self.mock_get_tp_size = self.tp_size_patcher.start()
        self.mock_get_tp_size.return_value = 2

    def tearDown(self):
        """Clean up patches."""
        self.dist_patcher.stop()
        self.tp_group_patcher.stop()
        self.tp_rank_patcher.stop()
        self.tp_size_patcher.stop()

    def test_manager_initializes_with_tp_awareness(self):
        """Test that manager correctly initializes TP rank/size."""
        from sglang.multimodal_gen.runtime.lora.lora_manager import DiffusionLoRAManager

        manager = DiffusionLoRAManager(max_loras_per_batch=8)

        self.assertEqual(manager.tp_rank, 0)
        self.assertEqual(manager.tp_size, 2)

    def test_sync_batch_info_broadcasts_in_tp_mode(self):
        """Test that batch info is broadcast from rank 0 in TP mode."""
        from sglang.multimodal_gen.runtime.lora.lora_manager import DiffusionLoRAManager

        mock_tp_group = MockTPGroup(rank=0, world_size=2)
        self.mock_get_tp_group.return_value = mock_tp_group

        manager = DiffusionLoRAManager(max_loras_per_batch=8)
        manager._tp_group = mock_tp_group

        active_loras = {"lora_a", "lora_b"}
        request_lora_map = {"req1": "lora_a", "req2": "lora_b"}
        lora_paths = {"lora_a": "/path/a", "lora_b": "/path/b"}

        result = manager._sync_batch_info_across_ranks(
            active_loras, request_lora_map, lora_paths
        )

        # Should have called broadcast_object
        self.assertEqual(len(mock_tp_group._broadcast_calls), 1)
        call = mock_tp_group._broadcast_calls[0]
        self.assertEqual(call["src"], 0)
        self.assertIn("active_loras", call["obj"])
        self.assertIn("request_lora_map", call["obj"])
        self.assertIn("lora_paths", call["obj"])

    def test_sync_batch_info_skipped_in_single_gpu(self):
        """Test that batch info sync is skipped for single GPU."""
        from sglang.multimodal_gen.runtime.lora.lora_manager import DiffusionLoRAManager

        # Configure as single GPU
        self.mock_get_tp_size.return_value = 1

        manager = DiffusionLoRAManager(max_loras_per_batch=8)
        manager.tp_size = 1

        active_loras = {"lora_a"}
        request_lora_map = {"req1": "lora_a"}
        lora_paths = {"lora_a": "/path/a"}

        result = manager._sync_batch_info_across_ranks(
            active_loras, request_lora_map, lora_paths
        )

        # Should return inputs unchanged without broadcast
        self.assertEqual(result[0], active_loras)
        self.assertEqual(result[1], request_lora_map)
        self.assertEqual(result[2], lora_paths)


class TestRowParallelLinearWithLoRAMultiGPU(unittest.TestCase):
    """Test RowParallelLinearWithLoRA's multi-GPU multi-LoRA handling."""

    def test_apply_multi_lora_does_not_allreduce_internally(self):
        """
        Test that RowParallelLinearWithLoRA._apply_multi_lora does NOT
        call all-reduce internally (it should be done in forward()).
        """
        from sglang.multimodal_gen.runtime.layers.lora.linear import (
            RowParallelLinearWithLoRA,
        )

        # Create a mock base layer
        mock_base_layer = MagicMock()
        mock_base_layer.input_size_per_partition = 64
        mock_base_layer.tp_size = 2
        mock_base_layer.weight = torch.randn(128, 64)

        # Create the LoRA wrapper
        lora_layer = RowParallelLinearWithLoRA(
            base_layer=mock_base_layer, lora_rank=16, lora_alpha=16
        )

        # Set up multi-LoRA
        lora_A = torch.randn(16, 128)  # (rank, in_features)
        lora_B = torch.randn(64, 16)  # (out_features, rank)

        lora_layer.lora_weights_pool = {"test_lora": (lora_A, lora_B)}
        lora_layer.lora_nickname_to_index = {"test_lora": 0}
        lora_layer.lora_adapter_configs = {"test_lora": {"alpha": 16.0, "rank": 16}}
        lora_layer.use_multi_lora = True
        lora_layer.active_lora_indices = torch.tensor([0, 0])
        lora_layer.disable_lora = False

        # Mock input and output
        x = torch.randn(2, 64)  # (batch, sharded_in_features)
        base_out = torch.randn(2, 64)  # (batch, out_features)

        # Patch all-reduce to track if it's called
        with patch(
            "sglang.multimodal_gen.runtime.layers.lora.linear.tensor_model_parallel_all_reduce"
        ) as mock_allreduce:
            mock_allreduce.side_effect = lambda x: x  # Identity

            result, _ = lora_layer._apply_multi_lora(x, base_out, None)

            # _apply_multi_lora should NOT call all-reduce
            # (the forward() method handles that)
            mock_allreduce.assert_not_called()


class TestColumnParallelLinearWithLoRASlicing(unittest.TestCase):
    """Test ColumnParallelLinearWithLoRA weight slicing for TP."""

    def test_slice_lora_b_weights_for_tp(self):
        """Test that LoRA B weights are correctly sliced for TP."""
        from sglang.multimodal_gen.runtime.layers.lora.linear import (
            ColumnParallelLinearWithLoRA,
        )

        # Mock base layer
        mock_base_layer = MagicMock()
        mock_base_layer.output_partition_sizes = [64]  # 64 per rank
        mock_base_layer.weight = torch.randn(64, 128)

        lora_layer = ColumnParallelLinearWithLoRA(
            base_layer=mock_base_layer, lora_rank=16, lora_alpha=16
        )

        # Full LoRA B: (total_out_features, rank)
        # With 2 ranks, total = 128
        lora_B = torch.randn(128, 16)

        # Test slicing for rank 0
        with patch(
            "sglang.multimodal_gen.runtime.layers.lora.linear.get_tp_rank"
        ) as mock_tp_rank:
            mock_tp_rank.return_value = 0
            sliced_B = lora_layer.slice_lora_b_weights(lora_B)
            self.assertEqual(sliced_B.shape, (64, 16))

            # Should be first 64 rows
            torch.testing.assert_close(sliced_B, lora_B[:64, :])

            # Test for rank 1
            mock_tp_rank.return_value = 1
            sliced_B = lora_layer.slice_lora_b_weights(lora_B)
            self.assertEqual(sliced_B.shape, (64, 16))

            # Should be second 64 rows
            torch.testing.assert_close(sliced_B, lora_B[64:128, :])


if __name__ == "__main__":
    unittest.main()
