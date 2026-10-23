import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn


class TestRingKVChunkHostSync(unittest.TestCase):
    def test_ring_hops_reuse_device_cu_seqlens(self):
        # every ring hop of every block used to upload [0, n] with a pageable,
        # host-blocking copy; the bounds are fixed per shape, so they upload once
        impl = flash_attn.FlashAttentionImpl(
            num_heads=2, head_size=8, causal=False, softmax_scale=0.5
        )
        query = torch.randn(6, 2, 8)
        key = torch.randn(4, 2, 8)
        calls = []

        def varlen(q, k, v, **kwargs):
            calls.append(kwargs)
            return q, torch.zeros(2, q.shape[0])

        with patch.object(flash_attn, "flash_attn_varlen_func", varlen):
            for _ in range(2):
                impl.forward_ring_kv_chunk(query, key, key)
        first, second = calls
        self.assertEqual(first["cu_seqlens_q"].tolist(), [0, 6])
        self.assertEqual(first["cu_seqlens_k"].tolist(), [0, 4])
        self.assertEqual(first["cu_seqlens_q"].dtype, torch.int32)
        self.assertIs(first["cu_seqlens_q"], second["cu_seqlens_q"])
        self.assertIs(first["cu_seqlens_k"], second["cu_seqlens_k"])


if __name__ == "__main__":
    unittest.main()
