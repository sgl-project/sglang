# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    Magi2BlockGrid,
    Magi2BlockGridAttention,
    cached_block_mask,
)

_HEADS = 4
_HEAD_DIM = 64
_LATENT_THW = (4, 8, 8)
_TAIL_ROOM = 256


def _grid(*, num_tail_tokens: int = 3, num_pad_tokens: int = 0) -> Magi2BlockGrid:
    return Magi2BlockGrid(
        latent_thw=_LATENT_THW,
        block_thw=(2, 2, 2),
        radius_thw=(1, 1, 1),
        num_tail_tokens=num_tail_tokens,
        num_pad_tokens=num_pad_tokens,
    )


def _bucketed_grid(*, num_tail_tokens: int) -> Magi2BlockGrid:
    """Same total length, different valid length: what bucketing produces."""
    return _grid(
        num_tail_tokens=num_tail_tokens,
        num_pad_tokens=_TAIL_ROOM - num_tail_tokens,
    )


def _qkv(grid: Magi2BlockGrid, *, device):
    generator = torch.Generator(device=device).manual_seed(0)
    return [
        torch.randn(grid.seq_len, _HEADS, _HEAD_DIM, device=device, generator=generator)
        for _ in range(3)
    ]


def _attend(attention, tensors, grid: Magi2BlockGrid) -> torch.Tensor:
    q, k, v = (t.transpose(0, 1)[None] for t in tensors)
    return attention._attend(q=q, k=k, v=v, grid=grid)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlexAttentionPath(unittest.TestCase):
    """Covers the compiled path; the mask tests only evaluate ``mask_mod``."""

    def setUp(self):
        self.device = torch.device("cuda")
        self.attention = Magi2BlockGridAttention(num_heads=_HEADS, head_dim=_HEAD_DIM)
        cached_block_mask.cache_clear()

    def test_sparse_block_mask_matches_the_dense_reference(self):
        # The tile occupancy handed to from_kv_blocks is a superset that mask_mod
        # refines, so too tight a superset silently drops keys.
        grid = _grid()
        tensors = _qkv(grid, device=self.device)
        dense = Magi2BlockGridAttention(
            num_heads=_HEADS, head_dim=_HEAD_DIM, dense_fallback=True
        )
        torch.testing.assert_close(
            _attend(self.attention, tensors, grid),
            _attend(dense, tensors, grid),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_full_blocks_match_the_dense_reference(self):
        # Block volume 8*4*4 = 128 is exactly one tile, so most allowed pairs are
        # fully unmasked and take from_kv_blocks' full_* path, which skips mask_mod
        # entirely. The other tests use an 8-token block, where no pair is ever full,
        # so without this one that path would be uncovered.
        grid = Magi2BlockGrid(
            latent_thw=(16, 16, 16),
            block_thw=(8, 4, 4),
            radius_thw=(2, 2, 2),
            num_tail_tokens=200,
        )
        mask = cached_block_mask(grid=grid, device=self.device)
        self.assertIsNotNone(mask.full_kv_num_blocks)
        self.assertGreater(int(mask.full_kv_num_blocks.sum()), 0)

        tensors = _qkv(grid, device=self.device)
        dense = Magi2BlockGridAttention(
            num_heads=_HEADS, head_dim=_HEAD_DIM, dense_fallback=True
        )
        torch.testing.assert_close(
            _attend(self.attention, tensors, grid),
            _attend(dense, tensors, grid),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_padded_rows_do_not_reach_a_valid_row(self):
        # Pad rows stay attendable as queries to avoid an all -inf softmax, so they
        # must not leak into the rows that are gathered back.
        plain = _grid()
        padded = _grid(num_pad_tokens=2)
        tensors = _qkv(plain, device=self.device)
        base = _attend(self.attention, tensors, plain)

        padded_tensors = [
            torch.cat([t, t[-1:].expand(2, *t.shape[1:])], dim=0) for t in tensors
        ]
        got = _attend(self.attention, padded_tensors, padded)

        self.assertFalse(torch.isnan(got).any())
        torch.testing.assert_close(
            got[:, :, : plain.seq_len], base, rtol=1e-4, atol=1e-4
        )

    def test_bucketed_lengths_share_one_compiled_graph(self):
        # dynamo guards a captured int by value, so build_mask_mod carries the valid
        # length as a tensor. Bucketing fixes the shape while the valid length still
        # varies per prompt; as an int each value here would be its own graph and
        # nine would trip the default recompile_limit of 8.
        import torch._dynamo as dynamo

        dynamo.reset()
        dynamo.utils.counters.clear()

        lengths = (3, 17, 64, 129, 200, 231, 244, 250, 255)
        seq_lens = set()
        for num_tail in lengths:
            grid = _bucketed_grid(num_tail_tokens=num_tail)
            seq_lens.add(grid.seq_len)
            _attend(self.attention, _qkv(grid, device=self.device), grid)

        # Guard the premise: the point is one shape with many valid lengths.
        self.assertEqual(len(seq_lens), 1)
        self.assertEqual(len(lengths), 9)
        self.assertEqual(dynamo.utils.counters["stats"]["unique_graphs"], 1)


if __name__ == "__main__":
    unittest.main()
