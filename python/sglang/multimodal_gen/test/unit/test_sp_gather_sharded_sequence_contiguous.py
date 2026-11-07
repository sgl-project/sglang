"""Regression test for ``USPAttention._gather_sharded_sequence`` contiguity.

The helper slices a replicated prefix/suffix off dim 1 and all-gathers the rest.
Slicing along dim 1 keeps the original row stride, so the shard is contiguous
only while the batch dim is 1 -- PyTorch ignores strides for size-1 dims. With a
batched CFG pass (B=2) both branches hand a non-contiguous view to
``all_gather_into_tensor``, which rejects it with "Tensors must be contiguous".

That is reachable from a default configuration: FLUX.2 on 2 GPUs auto-selects
``ulysses_degree=2`` and batches its two CFG branches, so the server fails during
warmup.

Single-process test: the all-gather is mocked so the slicing runs on CPU.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"
_SP = 2


class _CaptureAllGather:
    """Stand-in all-gather that records whether its input was contiguous."""

    def __init__(self):
        self.inputs = []

    def __call__(self, tensor, dim=-1):
        self.inputs.append(tensor)
        return torch.cat([tensor] * _SP, dim=dim)


class TestGatherShardedSequenceContiguous(unittest.TestCase):
    def _run(self, tensor, **kwargs):
        capture = _CaptureAllGather()
        with patch(f"{_LAYER}.sequence_model_parallel_all_gather", capture):
            out = USPAttention._gather_sharded_sequence(tensor, **kwargs)
        self.assertEqual(len(capture.inputs), 1)
        return out, capture.inputs[0]

    def test_prefix_shard_is_contiguous_when_batched(self):
        tensor = torch.randn(2, 16, 8)
        _, gathered_input = self._run(tensor, num_replicated_prefix=4)
        self.assertTrue(gathered_input.is_contiguous())

    def test_suffix_shard_is_contiguous_when_batched(self):
        tensor = torch.randn(2, 16, 8)
        _, gathered_input = self._run(tensor, num_replicated_suffix=4)
        self.assertTrue(gathered_input.is_contiguous())

    def test_batch_one_still_contiguous(self):
        # the case that happened to work before, and must keep working
        tensor = torch.randn(1, 16, 8)
        for kwargs in ({"num_replicated_prefix": 4}, {"num_replicated_suffix": 4}):
            with self.subTest(**kwargs):
                _, gathered_input = self._run(tensor, **kwargs)
                self.assertTrue(gathered_input.is_contiguous())

    def test_prefix_values_are_unchanged(self):
        tensor = torch.randn(2, 16, 8)
        out, gathered_input = self._run(tensor, num_replicated_prefix=4)
        torch.testing.assert_close(gathered_input, tensor[:, 4:])
        torch.testing.assert_close(out[:, :4], tensor[:, :4])
        self.assertEqual(out.shape, (2, 4 + 12 * _SP, 8))

    def test_suffix_values_are_unchanged(self):
        tensor = torch.randn(2, 16, 8)
        out, gathered_input = self._run(tensor, num_replicated_suffix=4)
        torch.testing.assert_close(gathered_input, tensor[:, :-4])
        torch.testing.assert_close(out[:, -4:], tensor[:, -4:])
        self.assertEqual(out.shape, (2, 12 * _SP + 4, 8))

    def test_no_replication_passes_the_tensor_through(self):
        tensor = torch.randn(2, 16, 8)
        _, gathered_input = self._run(tensor)
        self.assertIs(gathered_input, tensor)

    def test_prefix_and_suffix_together_are_rejected(self):
        with self.assertRaises(ValueError):
            USPAttention._gather_sharded_sequence(
                torch.randn(2, 16, 8),
                num_replicated_prefix=4,
                num_replicated_suffix=4,
            )


if __name__ == "__main__":
    unittest.main()
