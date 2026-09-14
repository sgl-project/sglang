"""Unit tests for managers/embed_types.py - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.test.test_utils import CustomTestCase


class TestPositionalEmbeds(CustomTestCase):
    def test_list_of_1d_tensors_stacked(self):
        """List of 1-D [hidden_dim] tensors is stacked into [N, hidden_dim]."""
        embeds = [torch.ones(4), torch.zeros(4)]
        pe = PositionalEmbeds(embeds=embeds, positions=[0, 1])
        self.assertEqual(tuple(pe.embeds.shape), (2, 4))

    def test_list_of_2d_tensors_concatenated(self):
        """List of [1, hidden_dim] tensors is concatenated into [N, hidden_dim]."""
        embeds = [torch.ones(1, 4), torch.zeros(1, 4)]
        pe = PositionalEmbeds(embeds=embeds, positions=[0, 1])
        self.assertEqual(tuple(pe.embeds.shape), (2, 4))

    def test_prestacked_tensor_accepted(self):
        """Pre-stacked [N, hidden_dim] tensor passes through to shape validation."""
        t = torch.ones(3, 4)
        pe = PositionalEmbeds(embeds=t, positions=[0, 1, 2])
        self.assertEqual(tuple(pe.embeds.shape), (3, 4))

    def test_single_1d_tensor_in_list(self):
        """Single 1-D tensor in list produces shape [1, hidden_dim]."""
        pe = PositionalEmbeds(embeds=[torch.ones(8)], positions=[5])
        self.assertEqual(tuple(pe.embeds.shape), (1, 8))

    def test_single_2d_tensor_in_list(self):
        """Single [1, hidden_dim] tensor in list produces shape [1, hidden_dim]."""
        pe = PositionalEmbeds(embeds=[torch.ones(1, 8)], positions=[5])
        self.assertEqual(tuple(pe.embeds.shape), (1, 8))

    def test_empty_list_raises(self):
        """Empty embeds list is rejected; torch.cat on an empty sequence raises."""
        with self.assertRaises((ValueError, RuntimeError)):
            PositionalEmbeds(embeds=[], positions=[])

    def test_count_mismatch_from_list_raises(self):
        """Embed count after stacking not matching positions length raises ValueError."""
        with self.assertRaises(ValueError):
            PositionalEmbeds(
                embeds=[torch.ones(4), torch.ones(4)],
                positions=[0],
            )

    def test_count_mismatch_from_tensor_raises(self):
        """Pre-stacked tensor row count not matching positions length raises ValueError."""
        with self.assertRaises(ValueError):
            PositionalEmbeds(embeds=torch.ones(3, 4), positions=[0, 1])

    def test_1d_stack_values_preserved(self):
        """Values in 1-D input tensors are preserved exactly after stacking."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        pe = PositionalEmbeds(embeds=[a, b], positions=[0, 1])
        self.assertTrue(torch.equal(pe.embeds[0], a))
        self.assertTrue(torch.equal(pe.embeds[1], b))

    def test_2d_cat_values_preserved(self):
        """Values in [1, hidden_dim] input tensors are preserved exactly after concat."""
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        pe = PositionalEmbeds(embeds=[a, b], positions=[0, 1])
        self.assertTrue(torch.equal(pe.embeds[0], a[0]))
        self.assertTrue(torch.equal(pe.embeds[1], b[0]))


if __name__ == "__main__":
    unittest.main()
