"""Per-item features must not carry the whole packed buffer over IPC.

`get_new_expanded_mm_items` splits one request-wide feature tensor into one
slice per placeholder. Those slices are views, and pickle -- which
`SGLANG_USE_PICKLE_IPC` selects by default -- serialises a view's entire
underlying storage rather than the view. Without narrowing, a request with N
images sends N x total_bytes to the scheduler.
"""

import pickle
import types
import unittest

import torch

from sglang.srt.managers.mm_utils import (
    TransportProxyTensor,
    narrow_mm_features_to_own_storage,
)
from sglang.srt.managers.mm_utils import narrow_value_to_own_storage as narrow
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _pickled(tensors) -> int:
    return sum(len(pickle.dumps(t, protocol=5)) for t in tensors)


def _own(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


class TestNarrowValueToOwnStorage(CustomTestCase):
    def test_a_view_of_a_packed_buffer_is_copied_out(self):
        packed = torch.randn(32, 16)
        view = packed[8:16]

        narrowed = narrow(view)

        self.assertTrue(torch.equal(narrowed, view))
        self.assertEqual(
            narrowed.untyped_storage().nbytes(),
            narrowed.numel() * narrowed.element_size(),
        )

    def test_a_tensor_that_owns_its_storage_is_returned_unchanged(self):
        """The single-image case must not pay a copy."""
        whole = torch.randn(32, 16)

        self.assertIs(narrow(whole), whole)

    def test_a_non_contiguous_view_is_narrowed_too(self):
        packed = torch.randn(32, 16)
        view = packed[:, :8]
        self.assertFalse(view.is_contiguous())

        narrowed = narrow(view)

        self.assertTrue(torch.equal(narrowed, view))
        self.assertEqual(
            narrowed.untyped_storage().nbytes(),
            narrowed.numel() * narrowed.element_size(),
        )

    def test_lists_and_tuples_are_narrowed_elementwise(self):
        packed = torch.randn(32, 16)
        for container in (list, tuple):
            with self.subTest(container=container.__name__):
                parts = container(packed[i * 8 : (i + 1) * 8] for i in range(4))
                out = narrow(parts)
                self.assertIsInstance(out, container)
                for narrowed, original in zip(out, parts):
                    self.assertTrue(torch.equal(narrowed, original))
                    self.assertEqual(
                        narrowed.untyped_storage().nbytes(),
                        narrowed.numel() * narrowed.element_size(),
                    )

    def test_non_tensor_values_pass_through(self):
        for value in (None, 7, "pixel_values", {"a": 1}):
            with self.subTest(value=value):
                self.assertIs(narrow(value), value)


class TestIpcCostOfASplitRequest(CustomTestCase):
    def test_split_views_cost_n_times_their_own_bytes_until_narrowed(self):
        # Large enough that pickle's per-tensor header is not part of the ratio.
        num_items = 8
        packed = torch.randn(num_items * 512, 1024)
        views = [packed[i * 512 : (i + 1) * 512] for i in range(num_items)]

        # What get_new_expanded_mm_items hands to transport today.
        self.assertAlmostEqual(_pickled(views) / _own(views), num_items, delta=0.01)

        narrowed = [narrow(v) for v in views]
        self.assertAlmostEqual(_pickled(narrowed) / _own(narrowed), 1.0, delta=0.01)
        for before, after in zip(views, narrowed):
            self.assertTrue(torch.equal(before, after))


class TestNarrowFeaturesOnItems(CustomTestCase):
    def _inputs(self, packed, num_items, rows):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=packed[i * rows : (i + 1) * rows],
                offsets=[(i * rows, (i + 1) * rows)],
            )
            for i in range(num_items)
        ]
        return types.SimpleNamespace(mm_items=items)

    def test_item_features_are_narrowed_in_place(self):
        packed = torch.randn(24, 32)
        mm_inputs = self._inputs(packed, 4, 6)
        before = [item.feature.clone() for item in mm_inputs.mm_items]

        narrow_mm_features_to_own_storage(mm_inputs)

        for item, original in zip(mm_inputs.mm_items, before):
            self.assertTrue(torch.equal(item.feature, original))
            self.assertEqual(
                item.feature.untyped_storage().nbytes(),
                item.feature.numel() * item.feature.element_size(),
            )

    def test_none_and_empty_are_tolerated(self):
        narrow_mm_features_to_own_storage(None)
        narrow_mm_features_to_own_storage(types.SimpleNamespace(mm_items=[]))
        narrow_mm_features_to_own_storage(types.SimpleNamespace())


class TestTransportSubclassesAreLeftAlone(CustomTestCase):
    """Cloning a TransportProxyTensor would drop the _metadata __new__ set."""

    def test_transport_proxy_tensor_is_not_cloned(self):
        packed = torch.randn(24, 32)
        proxy = TransportProxyTensor(packed[0:6], name="pixel_values")

        out = narrow(proxy)

        self.assertIs(out, proxy)
        self.assertEqual(out.name, "pixel_values")


if __name__ == "__main__":
    unittest.main()
