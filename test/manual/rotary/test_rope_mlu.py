"""Cambricon regression coverage for the packed FlagTree RoPE path."""

import unittest

import pytest
import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding


pytest.importorskip("torch_mlu")
if not torch.mlu.is_available():
    pytest.skip("Cambricon MLU is not available", allow_module_level=True)


class TestRotaryEmbeddingMLU(unittest.TestCase):
    def test_packed_rotary_matches_native(self):
        device = torch.device("mlu")
        rope = RotaryEmbedding(
            head_size=16,
            rotary_dim=16,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            dtype=torch.float16,
        ).to(device)
        positions = torch.tensor([4, 7, 1, 12, 3], device=device, dtype=torch.int32)
        query = torch.randn((5, 2 * 16), device=device, dtype=torch.float16)
        key = torch.randn((5, 16), device=device, dtype=torch.float16)

        q_ref, k_ref = rope.forward_native(positions, query.clone(), key.clone())
        q_mlu, k_mlu = rope.forward_mlu(positions, query.clone(), key.clone())

        torch.testing.assert_close(q_mlu, q_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k_mlu, k_ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
