# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    Magi2BlockGrid,
    build_mask_mod,
    cached_block_mask,
)


def _grid(num_tail: int, num_pad: int) -> Magi2BlockGrid:
    return Magi2BlockGrid(
        latent_thw=(4, 8, 8),
        block_thw=(2, 2, 2),
        radius_thw=(1, 1, 1),
        num_tail_tokens=num_tail,
        num_pad_tokens=num_pad,
    )


def _closed_over_ints(fn) -> list[int]:
    return [
        cell.cell_contents
        for cell in (fn.__closure__ or ())
        if isinstance(cell.cell_contents, int)
        and not isinstance(cell.cell_contents, bool)
    ]


class TestMaskModGuards(unittest.TestCase):
    def test_no_captured_int_varies_with_prompt_length(self):
        # dynamo guards on the value of a captured int, so any int that follows the
        # prompt recompiles flex_attention per prompt length and defeats the
        # sequence bucketing. Only the tier-fixed video token count may be an int.
        device = torch.device("cpu")
        seen = {
            tuple(
                _closed_over_ints(build_mask_mod(grid=_grid(t, 64 - t), device=device))
            )
            for t in (5, 17, 40)
        }
        self.assertEqual(len(seen), 1, f"captured ints vary with prompt length: {seen}")

    def test_valid_length_is_carried_as_a_tensor(self):
        mask_mod = build_mask_mod(grid=_grid(5, 59), device=torch.device("cpu"))
        tensors = [
            cell.cell_contents
            for cell in mask_mod.__closure__
            if isinstance(cell.cell_contents, torch.Tensor)
        ]
        # Bucket-shaped, so only its shape is guarded and the bucket pins that.
        self.assertTrue(any(t.dtype == torch.bool for t in tensors))

    def test_mask_still_bars_pad_as_a_key(self):
        grid = _grid(5, 59)
        mask_mod = build_mask_mod(grid=grid, device=torch.device("cpu"))
        positions = torch.arange(grid.seq_len)
        mask = mask_mod(
            None,
            None,
            positions[:, None].expand(grid.seq_len, grid.seq_len),
            positions[None, :].expand(grid.seq_len, grid.seq_len),
        )
        valid = grid.num_valid_tokens
        self.assertFalse(mask[:, valid:].any())
        self.assertTrue(mask[:valid, :valid].any())


class TestBlockMaskWidth(unittest.TestCase):
    def test_kv_indices_keeps_every_tile_column(self):
        # from_kv_blocks sizes its transposed buffer from kv_indices.shape[-1], so
        # trimming the column count to counts.max() cannot hold the tile ids stored
        # in it and asserts inside the index kernel.
        grid = _grid(0, 0)
        mask = cached_block_mask(grid=grid, device=torch.device("cpu"))
        self.assertEqual(mask.kv_indices.shape[-1], grid.num_tiles)


if __name__ == "__main__":
    unittest.main()
