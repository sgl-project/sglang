"""Unit tests for dtype preservation in topk_ids_logical_to_physical dispatch.

Verifies that _topk_ids_logical_to_physical_static and
_topk_ids_logical_to_physical_dynamic preserve the input topk_ids dtype
even when the dispatch map uses a wider dtype (int64). This matters for
backends like MORI that read expert indices via raw data_ptr() assuming
a specific dtype (int32).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    _topk_ids_logical_to_physical_dynamic,
    _topk_ids_logical_to_physical_static,
)
from sglang.test.test_utils import CustomTestCase

NUM_LOGICAL = 256
NUM_PHYSICAL = 256


def _make_identity_info():
    """Create dispatch info with an int64 identity map (logical == physical)."""
    dispatch_map = torch.arange(NUM_LOGICAL, dtype=torch.int64)
    all_physical_map = torch.arange(NUM_LOGICAL, dtype=torch.int64).unsqueeze(1)
    num_valid = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    return ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="static",
        partial_logical_to_rank_dispatch_physical_map=dispatch_map,
        partial_logical_to_all_physical_map=all_physical_map,
        partial_logical_to_all_physical_map_num_valid=num_valid,
        num_physical_experts=NUM_PHYSICAL,
    )


def _make_permuted_info():
    """Create dispatch info with a non-trivial int64 permutation."""
    perm = torch.randperm(NUM_LOGICAL, dtype=torch.int64)
    all_physical_map = perm.unsqueeze(1)
    num_valid = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    return ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="static",
        partial_logical_to_rank_dispatch_physical_map=perm,
        partial_logical_to_all_physical_map=all_physical_map,
        partial_logical_to_all_physical_map_num_valid=num_valid,
        num_physical_experts=NUM_PHYSICAL,
    )


class TestStaticDispatchDtype(CustomTestCase):
    """Tests for _topk_ids_logical_to_physical_static dtype preservation."""

    def test_preserves_int32_dtype_on_hip(self):
        """int32 input must produce int32 output when dispatch map is int64."""
        info = _make_identity_info()
        topk_ids = torch.tensor([5, 103, 206], dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_static(topk_ids, info)
        self.assertEqual(result.dtype, torch.int32)

    def test_preserves_int64_dtype(self):
        """int64 input with int64 map should stay int64."""
        info = _make_identity_info()
        topk_ids = torch.tensor([5, 103, 206], dtype=torch.int64)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_static(topk_ids, info)
        self.assertEqual(result.dtype, torch.int64)

    def test_values_correct_after_cast(self):
        """Remapped values are correct, not corrupted by the dtype cast."""
        info = _make_permuted_info()
        topk_ids = torch.tensor([0, 1, 2, 127, 255], dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_static(topk_ids, info)
        expected = info.partial_logical_to_rank_dispatch_physical_map[topk_ids.long()]
        self.assertTrue(torch.equal(result, expected.to(torch.int32)))

    def test_batch_shape_preserved(self):
        """2-D input shape is preserved through the remap."""
        info = _make_identity_info()
        topk_ids = torch.randint(0, NUM_LOGICAL, (32, 8), dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_static(topk_ids, info)
        self.assertEqual(result.shape, (32, 8))
        self.assertEqual(result.dtype, torch.int32)


class TestDynamicDispatchDtype(CustomTestCase):
    """Tests for _topk_ids_logical_to_physical_dynamic dtype preservation."""

    def test_preserves_int32_dtype_on_hip(self):
        """int32 input must produce int32 output when dispatch map is int64."""
        info = _make_permuted_info()
        info.ep_dispatch_algorithm = "dynamic"
        topk_ids = torch.tensor([5, 103, 206], dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
        self.assertEqual(result.dtype, torch.int32)

    def test_preserves_int64_dtype(self):
        """int64 input with int64 map should stay int64."""
        info = _make_permuted_info()
        info.ep_dispatch_algorithm = "dynamic"
        topk_ids = torch.tensor([5, 103, 206], dtype=torch.int64)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
        self.assertEqual(result.dtype, torch.int64)

    def test_values_correct_single_candidate(self):
        """With one candidate per logical expert, dynamic matches static."""
        info = _make_permuted_info()
        info.ep_dispatch_algorithm = "dynamic"
        topk_ids = torch.tensor([0, 1, 2, 127, 255], dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
        expected = info.partial_logical_to_all_physical_map[topk_ids.long(), 0]
        self.assertTrue(torch.equal(result, expected.to(torch.int32)))

    def test_batch_shape_preserved(self):
        """2-D input shape is preserved through the remap."""
        info = _make_permuted_info()
        info.ep_dispatch_algorithm = "dynamic"
        topk_ids = torch.randint(0, NUM_LOGICAL, (32, 8), dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
        self.assertEqual(result.shape, (32, 8))
        self.assertEqual(result.dtype, torch.int32)

    def test_no_op_dtype_bug_regression(self):
        """Regression test: the old code did topk_ids.to(topk_ids.dtype) after
        overwriting topk_ids with the int64 lookup result, which was a no-op.
        Verify the output is actually int32, not silently int64."""
        info = _make_permuted_info()
        info.ep_dispatch_algorithm = "dynamic"
        topk_ids = torch.randint(0, NUM_LOGICAL, (64, 8), dtype=torch.int32)
        with patch("sglang.srt.eplb.expert_location_dispatch._is_hip", True):
            result = _topk_ids_logical_to_physical_dynamic(topk_ids, info)
        self.assertEqual(
            result.dtype,
            torch.int32,
            "Dynamic dispatch returned int64 — the no-op dtype cast bug has regressed",
        )
        self.assertEqual(result.element_size(), 4)


if __name__ == "__main__":
    unittest.main()
