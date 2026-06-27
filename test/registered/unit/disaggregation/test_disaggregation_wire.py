import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_list,
    pack_int_lists,
    pack_list_of_buffers,
    pack_nested_transfer_layout,
    pack_transfer_layout,
    unpack_int_list,
    unpack_int_lists,
    unpack_list_of_buffers,
    unpack_nested_transfer_layout,
    unpack_transfer_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(CustomTestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_flat_int_list_roundtrip(self):
        self.assertEqual(unpack_int_list(pack_int_list([7, 8, 9], "I"), "I"), [7, 8, 9])
        self.assertEqual(pack_int_list([], "I"), b"")
        self.assertEqual(unpack_int_list(b"", "I"), [])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestCpKvLayerSplitTransferLayoutWire(CustomTestCase):
    def test_layer_split_transfer_layout_roundtrip_preserves_none_slots(self):
        layout = [("dsv4_c4_kv", 1), None, ("dsv4_c128_kv", 3)]

        self.assertEqual(unpack_transfer_layout(pack_transfer_layout(layout)), layout)
        self.assertEqual(pack_transfer_layout([]), b"")
        self.assertEqual(unpack_transfer_layout(b""), [])

    def test_layer_split_state_layout_roundtrip_preserves_component_boundaries(self):
        layouts = [
            [("dsv4_swa_kv", 0), None],
            [("dsv4_attention_state", 1), ("dsv4_indexer_state", 1)],
        ]

        self.assertEqual(
            unpack_nested_transfer_layout(pack_nested_transfer_layout(layouts)),
            layouts,
        )
        self.assertEqual(pack_nested_transfer_layout([]), b"")
        self.assertEqual(unpack_nested_transfer_layout(b""), [])


class TestCpKvLayerSplitDescriptorMatching(CustomTestCase):
    def _build_params(self, **kwargs):
        mgr = SimpleNamespace(cp_kv_layer_split=kwargs.pop("cp_kv_layer_split", True))
        return CommonKVManager.build_descriptor_matched_transfer_params(mgr, **kwargs)

    def test_descriptor_matching_checks_destination_item_size(self):
        with self.assertRaisesRegex(RuntimeError, "item size mismatch"):
            self._build_params(
                src_data_ptrs=[100],
                dst_data_ptrs=[200],
                item_lens=[16],
                src_data_layout=[("dsv4_c4_kv", 1)],
                dst_data_layout=[("dsv4_c4_kv", 1)],
                dst_item_lens=[32],
            )

    def test_descriptor_matching_returns_pointer_item_len_tuples(self):
        self.assertEqual(
            self._build_params(
                src_data_ptrs=[100],
                dst_data_ptrs=[200],
                item_lens=[16],
                src_data_layout=[("dsv4_c4_kv", 1)],
                dst_data_layout=[("dsv4_c4_kv", 1)],
                dst_item_lens=[16],
            ),
            [(100, 200, 16)],
        )


class TestGroupConcurrentContiguous(CustomTestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_both_empty(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([])), ([], [])
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


if __name__ == "__main__":
    unittest.main()
