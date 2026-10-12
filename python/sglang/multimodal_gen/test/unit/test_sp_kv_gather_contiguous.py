"""Sequence-parallel K/V gather must materialize non-contiguous views.

Flux2 packs Q, K, and V with one projection and then ``chunk``s them. That
view is not contiguous, and ``all_gather_into_tensor`` rejects it.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.test.test_utils import CustomTestCase


class TestSequenceParallelKvGatherContiguous(CustomTestCase):
    def test_noncontiguous_kv_chunk_is_materialized_before_all_gather(self):
        packed = torch.randn(2, 8, 12)
        key = packed.chunk(3, dim=-1)[1]
        self.assertFalse(key.is_contiguous())

        captured = {}

        def fake_all_gather(tensor, dim=-1):
            captured["tensor"] = tensor
            captured["dim"] = dim
            return tensor

        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.sequence_model_parallel_all_gather",
            fake_all_gather,
        ):
            gathered = USPAttention._gather_sharded_sequence(key)

        self.assertEqual(captured["dim"], 1)
        self.assertTrue(captured["tensor"].is_contiguous())
        self.assertTrue(torch.equal(captured["tensor"], key))
        self.assertTrue(torch.equal(gathered, key))


if __name__ == "__main__":
    unittest.main()
