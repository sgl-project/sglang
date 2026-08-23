# SPDX-License-Identifier: Apache-2.0
"""Tile noise must not depend on which rank decodes the tile.

The tiles of a diffusion decode are independent, so they can be split across
sequence-parallel ranks. That only holds if each tile's noise is a function of
the tile itself: drawing from one shared generator inside the loop ties the
result to the order tiles are visited, and a rank computing a subset would then
see different noise than a single rank does.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
    _tile_generator,
    _tile_intervals,
)


class TestTileGenerator(unittest.TestCase):
    def _draw(self, base_seed, index, shape=(2, 3)):
        return torch.randn(
            shape, generator=_tile_generator(base_seed, index, torch.device("cpu"))
        )

    def test_a_tile_draws_the_same_noise_however_it_is_reached(self):
        # The point of the helper: index 5 gives the same noise whether it was
        # the first tile this rank computed or the fifth.
        first = self._draw(42, 5)
        again = self._draw(42, 5)
        torch.testing.assert_close(first, again)

    def test_neighbouring_tiles_get_different_noise(self):
        a, b = self._draw(42, 0), self._draw(42, 1)
        self.assertFalse(torch.allclose(a, b))

    def test_the_request_seed_still_changes_the_result(self):
        a, b = self._draw(42, 3), self._draw(43, 3)
        self.assertFalse(torch.allclose(a, b))

    def test_a_round_robin_split_covers_every_tile_exactly_once(self):
        # How `tiled_decode` assigns tiles: rank r takes r, r+W, r+2W, ...
        for total in (1, 7, 14, 15):
            for world_size in (1, 2, 3, 4):
                assigned = [
                    i
                    for rank in range(world_size)
                    for i in range(rank, total, world_size)
                ]
                self.assertEqual(sorted(assigned), list(range(total)))


class TestTileIntervals(unittest.TestCase):
    def test_intervals_cover_the_axis(self):
        intervals = _tile_intervals(30, 12, 8, 4)
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], 30)

    def test_a_short_remnant_is_merged_into_the_previous_tile(self):
        # 25 with stride 8 would leave a 1-long trailing tile, which is below
        # the neighborhood kernel and cannot be decoded on its own.
        for start, end in _tile_intervals(25, 12, 8, 4):
            self.assertGreaterEqual(end - start, 4)

    def test_an_axis_shorter_than_one_tile_stays_whole(self):
        self.assertEqual(_tile_intervals(5, 12, 8, 4), [(0, 5)])


if __name__ == "__main__":
    unittest.main()
