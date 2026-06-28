"""Tests for sglang.srt.kv_canary.expected_inputs: ExpectedInputs allocation and slicing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.test.test_utils import CustomTestCase

_DEVICE = torch.device("cpu")


class TestExpectedInputsAllocate(CustomTestCase):
    def test_tokens_shape(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        self.assertEqual(ei.tokens.shape, torch.Size([8]))

    def test_tokens_dtype(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        self.assertEqual(ei.tokens.dtype, torch.int64)

    def test_positions_shape(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        self.assertEqual(ei.positions.shape, torch.Size([8]))

    def test_positions_dtype(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        self.assertEqual(ei.positions.dtype, torch.int64)

    def test_capacity_one(self):
        ei = ExpectedInputs.allocate(capacity=1, device=_DEVICE)
        self.assertEqual(ei.tokens.shape, torch.Size([1]))
        self.assertEqual(ei.positions.shape, torch.Size([1]))

    def test_tokens_and_positions_are_separate_tensors(self):
        ei = ExpectedInputs.allocate(capacity=4, device=_DEVICE)
        self.assertIsNot(ei.tokens, ei.positions)

    def test_frozen_dataclass_rejects_mutation(self):
        ei = ExpectedInputs.allocate(capacity=4, device=_DEVICE)
        with self.assertRaises((AttributeError, TypeError)):
            ei.tokens = torch.zeros(4, dtype=torch.int64)


class TestExpectedInputsSlice(CustomTestCase):
    def test_slice_shape(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        s = ei.slice(3)
        self.assertEqual(s.tokens.shape, torch.Size([3]))
        self.assertEqual(s.positions.shape, torch.Size([3]))

    def test_slice_is_view_of_original_tokens(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        s = ei.slice(3)
        self.assertEqual(s.tokens.data_ptr(), ei.tokens.data_ptr())

    def test_slice_is_view_of_original_positions(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        s = ei.slice(3)
        self.assertEqual(s.positions.data_ptr(), ei.positions.data_ptr())

    def test_slice_zero_returns_empty(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        s = ei.slice(0)
        self.assertEqual(s.tokens.numel(), 0)
        self.assertEqual(s.positions.numel(), 0)

    def test_slice_full_capacity_same_as_original(self):
        ei = ExpectedInputs.allocate(capacity=5, device=_DEVICE)
        s = ei.slice(5)
        self.assertEqual(s.tokens.shape, ei.tokens.shape)
        self.assertEqual(s.positions.shape, ei.positions.shape)

    def test_slice_data_shared_with_original(self):
        ei = ExpectedInputs.allocate(capacity=8, device=_DEVICE)
        ei.tokens.fill_(42)
        s = ei.slice(3)
        self.assertEqual(s.tokens.tolist(), [42, 42, 42])

    def test_slice_returns_expected_inputs_instance(self):
        ei = ExpectedInputs.allocate(capacity=4, device=_DEVICE)
        s = ei.slice(2)
        self.assertIsInstance(s, ExpectedInputs)

    def test_slice_dtype_preserved(self):
        ei = ExpectedInputs.allocate(capacity=4, device=_DEVICE)
        s = ei.slice(2)
        self.assertEqual(s.tokens.dtype, torch.int64)
        self.assertEqual(s.positions.dtype, torch.int64)

    def test_writes_through_slice_visible_in_original(self):
        ei = ExpectedInputs.allocate(capacity=6, device=_DEVICE)
        ei.tokens.zero_()
        s = ei.slice(3)
        s.tokens.fill_(7)
        self.assertEqual(ei.tokens[:3].tolist(), [7, 7, 7])


if __name__ == "__main__":
    unittest.main(verbosity=3)
