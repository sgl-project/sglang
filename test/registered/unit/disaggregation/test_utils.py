"""Unit tests for srt/disaggregation/utils."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    ReqToMetadataIdxAllocator,
    append_state_component,
    filter_kv_indices_for_cp_rank,
    page_indices_to_cp_rank_page_indices,
)
from sglang.test.test_utils import CustomTestCase


class TestPageIndicesToCpRankPageIndices(CustomTestCase):
    """Tests for page_indices_to_cp_rank_page_indices."""

    def test_cp_size_one_is_passthrough(self):
        page_indices = np.array([3, 4, 5])
        result = page_indices_to_cp_rank_page_indices(
            page_indices=page_indices, total_pages=3, cp_rank=0, cp_size=1
        )
        # cp_size <= 1 returns the exact input array.
        self.assertIs(result, page_indices)

    def test_empty_array_returns_empty(self):
        page_indices = np.array([], dtype=np.int64)
        result = page_indices_to_cp_rank_page_indices(
            page_indices=page_indices, total_pages=0, cp_rank=1, cp_size=4
        )
        self.assertEqual(result.size, 0)

    def test_even_split_exact_local_indices(self):
        # total_pages=8, cp_size=2 -> each rank owns 4 contiguous pages.
        page_indices = np.arange(8)
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=8, cp_rank=0, cp_size=2
            ),
            np.array([0, 1, 2, 3]),
        )
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=8, cp_rank=1, cp_size=2
            ),
            np.array([4, 5, 6, 7]),
        )

    def test_uneven_split_remainder_distribution(self):
        # total_pages=7, cp_size=3 -> remainder 1 goes to rank 0 (min(cp_rank, rem)).
        page_indices = np.arange(7)
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=7, cp_rank=0, cp_size=3
            ),
            np.array([0, 1, 2]),
        )
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=7, cp_rank=1, cp_size=3
            ),
            np.array([3, 4]),
        )
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=7, cp_rank=2, cp_size=3
            ),
            np.array([5, 6]),
        )

    def test_global_to_local_mapping_with_offset(self):
        # Pages live at a non-zero first_page; local slices map back to globals.
        page_indices = np.arange(100, 106)
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=6, cp_rank=0, cp_size=3
            ),
            np.array([100, 101]),
        )
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=6, cp_rank=1, cp_size=3
            ),
            np.array([102, 103]),
        )
        np.testing.assert_array_equal(
            page_indices_to_cp_rank_page_indices(
                page_indices=page_indices, total_pages=6, cp_rank=2, cp_size=3
            ),
            np.array([104, 105]),
        )

    def test_rank_beyond_total_pages_returns_empty(self):
        # total_pages=2, cp_size=4 -> ranks 2 and 3 own no pages.
        page_indices = np.array([10, 11])
        result = page_indices_to_cp_rank_page_indices(
            page_indices=page_indices, total_pages=2, cp_rank=3, cp_size=4
        )
        self.assertEqual(result.size, 0)


class TestReqToMetadataIdxAllocator(CustomTestCase):
    """Tests for ReqToMetadataIdxAllocator."""

    def test_initial_pool_size(self):
        allocator = ReqToMetadataIdxAllocator(4)
        self.assertEqual(allocator.size, 4)
        self.assertEqual(allocator.available_size(), 4)

    def test_alloc_returns_fifo_order(self):
        allocator = ReqToMetadataIdxAllocator(3)
        self.assertEqual(allocator.alloc(), 0)
        self.assertEqual(allocator.alloc(), 1)
        self.assertEqual(allocator.alloc(), 2)

    def test_alloc_returns_none_when_exhausted(self):
        allocator = ReqToMetadataIdxAllocator(2)
        allocator.alloc()
        allocator.alloc()
        self.assertIsNone(allocator.alloc())
        self.assertEqual(allocator.available_size(), 0)

    def test_free_then_alloc_returns_freed_index(self):
        allocator = ReqToMetadataIdxAllocator(2)
        allocator.alloc()
        idx = allocator.alloc()
        allocator.free(idx)
        self.assertEqual(allocator.alloc(), idx)

    def test_multiple_free_alloc_cycles_maintain_state(self):
        allocator = ReqToMetadataIdxAllocator(3)
        a = allocator.alloc()  # 0
        b = allocator.alloc()  # 1
        allocator.free(a)
        allocator.free(b)
        # Remaining free slot (2) is dequeued before the freed ones (FIFO).
        self.assertEqual(allocator.alloc(), 2)
        self.assertEqual(allocator.alloc(), a)
        self.assertEqual(allocator.alloc(), b)
        self.assertIsNone(allocator.alloc())


class TestDisaggregationMode(CustomTestCase):
    """Tests for the DisaggregationMode enum."""

    def test_to_engine_type(self):
        self.assertEqual(
            DisaggregationMode.to_engine_type(DisaggregationMode.PREFILL.value),
            "prefill",
        )
        self.assertEqual(
            DisaggregationMode.to_engine_type(DisaggregationMode.DECODE.value),
            "decode",
        )
        # Anything that is not prefill/decode (e.g. NULL) maps to "unified".
        self.assertEqual(
            DisaggregationMode.to_engine_type(DisaggregationMode.NULL.value),
            "unified",
        )

    def test_all_members_accessible(self):
        self.assertEqual(DisaggregationMode.NULL.value, "null")
        self.assertEqual(DisaggregationMode.PREFILL.value, "prefill")
        self.assertEqual(DisaggregationMode.DECODE.value, "decode")
        self.assertEqual(
            {member.name for member in DisaggregationMode},
            {"NULL", "PREFILL", "DECODE"},
        )


class TestFilterKvIndicesForCpRank(CustomTestCase):
    """Tests for filter_kv_indices_for_cp_rank."""

    @staticmethod
    def _stub_kv_mgr(cp_rank, cp_size):
        # Minimal stub: only the CP rank/size attributes are read.
        return SimpleNamespace(attn_cp_rank=cp_rank, attn_cp_size=cp_size)

    def test_cp_size_one_returns_all_indices(self):
        kv_indices = np.array([5, 6, 7, 8])
        index_slice = slice(0, 4)
        new_indices, new_slice = filter_kv_indices_for_cp_rank(
            self._stub_kv_mgr(0, 1), kv_indices, index_slice
        )
        np.testing.assert_array_equal(new_indices, kv_indices)
        self.assertEqual(new_slice, slice(0, 4))

    def test_subset_per_rank_with_cp_size_two(self):
        kv_indices = np.array([0, 1, 2, 3])
        index_slice = slice(10, 14)

        new_indices, new_slice = filter_kv_indices_for_cp_rank(
            self._stub_kv_mgr(0, 2), kv_indices, index_slice
        )
        np.testing.assert_array_equal(new_indices, np.array([0, 1]))
        self.assertEqual(new_slice, slice(10, 12))

        new_indices, new_slice = filter_kv_indices_for_cp_rank(
            self._stub_kv_mgr(1, 2), kv_indices, index_slice
        )
        np.testing.assert_array_equal(new_indices, np.array([2, 3]))
        self.assertEqual(new_slice, slice(12, 14))


class TestAppendStateComponent(CustomTestCase):
    """Tests for append_state_component."""

    @staticmethod
    def _stub_kv_args():
        # Minimal stub exposing the five state-component lists the helper appends to.
        return SimpleNamespace(
            state_types=[],
            state_data_ptrs=[],
            state_data_lens=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
        )

    def test_append_to_empty_lists(self):
        kv_args = self._stub_kv_args()
        append_state_component(kv_args, StateType.MAMBA, [1, 2], [3, 4], [5, 6], [7])
        self.assertEqual(kv_args.state_types, [StateType.MAMBA])
        self.assertEqual(kv_args.state_data_ptrs, [[1, 2]])
        self.assertEqual(kv_args.state_data_lens, [[3, 4]])
        self.assertEqual(kv_args.state_item_lens, [[5, 6]])
        self.assertEqual(kv_args.state_dim_per_tensor, [[7]])

    def test_append_multiple_components_and_default_dim(self):
        kv_args = self._stub_kv_args()
        append_state_component(kv_args, StateType.MAMBA, [1], [2], [3], [4])
        # dim_per_tensor defaults to an empty list when omitted.
        append_state_component(kv_args, StateType.SWA, [8], [9], [10])
        self.assertEqual(kv_args.state_types, [StateType.MAMBA, StateType.SWA])
        self.assertEqual(kv_args.state_data_ptrs, [[1], [8]])
        self.assertEqual(kv_args.state_data_lens, [[2], [9]])
        self.assertEqual(kv_args.state_item_lens, [[3], [10]])
        self.assertEqual(kv_args.state_dim_per_tensor, [[4], []])


if __name__ == "__main__":
    unittest.main()
