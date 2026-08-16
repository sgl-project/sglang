# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    Magi2BlockGrid,
    block_scan_order,
    build_mask_mod,
)


def _expected_allowed(grid: Magi2BlockGrid, order: torch.Tensor) -> torch.Tensor:
    """Ground-truth neighborhood, derived from raster coordinates.

    ``order`` maps a position in the permuted sequence back to its raster token,
    so the block a token belongs to is computed from that raster index rather
    than from its position, which is what the mask does.
    """
    t_dim, h_dim, w_dim = grid.latent_thw
    block_t, block_h, block_w = grid.block_thw
    seq = grid.seq_len

    allowed = torch.zeros(seq, seq, dtype=torch.bool)
    coords = []
    for position in range(seq):
        raster = int(order[position])
        if raster >= grid.num_video_tokens:
            coords.append(None)
            continue
        t = raster // (h_dim * w_dim)
        h = raster // w_dim % h_dim
        w = raster % w_dim
        coords.append((t // block_t, h // block_h, w // block_w))

    for q in range(seq):
        for kv in range(seq):
            if coords[q] is None or coords[kv] is None:
                allowed[q, kv] = True
                continue
            allowed[q, kv] = all(
                abs(a - b) <= r
                for a, b, r in zip(coords[q], coords[kv], grid.radius_thw)
            )
    return allowed


def _mask_matrix(grid: Magi2BlockGrid) -> torch.Tensor:
    mask_mod = build_mask_mod(grid=grid, device=torch.device("cpu"))
    positions = torch.arange(grid.seq_len)
    q_idx = positions[:, None].expand(grid.seq_len, grid.seq_len)
    kv_idx = positions[None, :].expand(grid.seq_len, grid.seq_len)
    return mask_mod(None, None, q_idx, kv_idx)


class TestBlockScanOrder(unittest.TestCase):
    def setUp(self):
        # The block grid has to be wider than the radius on some axis, or every
        # block is a neighbor of every other and the mask is all-True, which makes
        # any ordering look correct.
        self.grid = Magi2BlockGrid(
            latent_thw=(4, 8, 8),
            block_thw=(2, 2, 2),
            radius_thw=(1, 1, 1),
            num_tail_tokens=3,
        )

    def assert_restrictive(self, mask: torch.Tensor) -> None:
        video = self.grid.num_video_tokens
        self.assertTrue(mask[:video, :video].any())
        self.assertFalse(mask[:video, :video].all())

    def test_permutation_is_a_bijection_with_the_tail_appended(self):
        order, restore = block_scan_order(grid=self.grid, device=torch.device("cpu"))
        self.assertEqual(order.numel(), self.grid.seq_len)
        self.assertEqual(sorted(order.tolist()), list(range(self.grid.seq_len)))
        self.assertTrue(torch.equal(order[restore], torch.arange(order.numel())))
        # The tail must stay put: only video tokens live on the block grid.
        tail = torch.arange(self.grid.num_video_tokens, self.grid.seq_len)
        self.assertTrue(torch.equal(order[-tail.numel() :], tail))

    def test_mask_matches_the_grid_neighborhood_in_scan_order(self):
        order, _ = block_scan_order(grid=self.grid, device=torch.device("cpu"))
        mask = _mask_matrix(self.grid)
        self.assert_restrictive(mask)
        self.assertTrue(torch.equal(mask, _expected_allowed(self.grid, order)))

    def test_mask_is_wrong_without_the_permutation(self):
        # The mask assigns a token's block from its position, so feeding it a
        # raster-ordered sequence silently attends over the wrong neighborhood.
        # This is the failure the permutation exists to prevent.
        identity = torch.arange(self.grid.seq_len)
        mask = _mask_matrix(self.grid)
        self.assert_restrictive(mask)
        self.assertFalse(torch.equal(mask, _expected_allowed(self.grid, identity)))

    def test_shard_padding_is_not_attendable(self):
        # Pad rows repeat the last real row, and the tail is globally visible, so
        # leaving them unmasked lets every query attend to a duplicate.
        padded = Magi2BlockGrid(
            latent_thw=(4, 8, 8),
            block_thw=(2, 2, 2),
            radius_thw=(1, 1, 1),
            num_tail_tokens=3,
            num_pad_tokens=2,
        )
        mask = _mask_matrix(padded)
        valid = padded.num_valid_tokens
        self.assertEqual(mask.shape[0], valid + 2)
        self.assertFalse(mask[:, valid:].any())
        # Pad rows keep their own keys: a fully masked query row softmaxes over
        # all -inf and returns NaN. Their outputs are dropped on gather instead.
        self.assertTrue(mask[valid:, :valid].any())
        self.assertTrue(torch.equal(mask[:valid, :valid], _mask_matrix(self.grid)))

    def test_factory_forwards_the_pad_count(self):
        # Built through from_arch_config, not the struct: dropping num_pad_tokens
        # here sizes the mask short of the tensor, and flex_attention rejects it
        # only for prompts whose token count is not a multiple of the world size.
        from sglang.multimodal_gen.configs.models.dits.magi2 import (
            Magi2RefinerArchConfig,
        )

        grid = Magi2BlockGrid.from_arch_config(
            arch_config=Magi2RefinerArchConfig(),
            latent_thw=(4, 8, 8),
            num_tail_tokens=3,
            num_pad_tokens=2,
        )
        self.assertEqual(grid.num_pad_tokens, 2)
        self.assertEqual(grid.seq_len, grid.num_valid_tokens + 2)


if __name__ == "__main__":
    unittest.main()
