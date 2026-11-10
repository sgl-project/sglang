"""Cambricon regression coverage for the FlagTree RoPE path."""

import unittest

import pytest
import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding

pytest.importorskip("torch_mlu")
if not torch.mlu.is_available():
    pytest.skip("Cambricon MLU is not available", allow_module_level=True)


class TestRotaryEmbeddingMLU(unittest.TestCase):
    def _check(self, positions, *, head_size, rotary_dim, max_pos, is_neox_style):
        device = torch.device("mlu")
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_pos,
            base=10000,
            is_neox_style=is_neox_style,
            dtype=torch.float16,
        ).to(device)
        positions = positions.to(device)
        num_tokens = positions.numel()
        query = torch.randn(
            (num_tokens, 4 * head_size), device=device, dtype=torch.float16
        )
        key = torch.randn(
            (num_tokens, 2 * head_size), device=device, dtype=torch.float16
        )

        q_ref, k_ref = rope.forward_native(positions, query.clone(), key.clone())
        q_mlu, k_mlu = rope.forward_mlu(positions, query.clone(), key.clone())

        torch.testing.assert_close(q_mlu, q_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k_mlu, k_ref, atol=2e-2, rtol=2e-2)

    def test_rotary_matches_native(self):
        positions = torch.tensor([4, 7, 1, 12, 3], dtype=torch.int64)
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                self._check(
                    positions,
                    head_size=16,
                    rotary_dim=16,
                    max_pos=32,
                    is_neox_style=is_neox_style,
                )

    def test_non_contiguous_positions(self):
        # Several requests concatenated, each starting at a different position.
        positions = torch.cat(
            (
                torch.arange(37, 337),
                torch.arange(0, 512),
                torch.arange(900, 1101),
                torch.randint(0, 2048, (64,)),
            )
        )
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                self._check(
                    positions,
                    head_size=128,
                    rotary_dim=128,
                    max_pos=2048,
                    is_neox_style=is_neox_style,
                )

    def test_partial_rotary(self):
        positions = torch.randint(0, 1024, (257,))
        self._check(
            positions, head_size=128, rotary_dim=64, max_pos=1024, is_neox_style=True
        )

    def test_more_tokens_than_cache_rows(self):
        # Two 1024-token requests with a 1032-row cos/sin cache.
        positions = torch.cat((torch.arange(1024), torch.arange(1024)))
        self._check(
            positions, head_size=128, rotary_dim=128, max_pos=1032, is_neox_style=True
        )


if __name__ == "__main__":
    unittest.main()
