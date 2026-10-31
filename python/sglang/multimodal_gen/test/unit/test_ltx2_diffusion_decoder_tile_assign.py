import unittest

from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
    _assign_tiles_lpt,
)
from sglang.test.test_utils import CustomTestCase


class TestLTX2DiffusionDecoderTileAssign(CustomTestCase):
    """Longest-processing-time tile->rank assignment for parallel decode.

    The assignment must be deterministic (every rank computes it independently
    with no communication) and balance decode load across ranks.
    """

    def _loads(self, volumes, owners, world_size):
        loads = [0] * world_size
        for vol, rank in zip(volumes, owners):
            loads[rank] += vol
        return loads

    def test_single_rank_all_local(self):
        volumes = [5, 3, 9, 1]
        self.assertEqual(_assign_tiles_lpt(volumes, 1), [0, 0, 0, 0])

    def test_all_tiles_assigned_to_valid_ranks(self):
        volumes = [7, 2, 5, 4, 9, 1, 6, 3]
        owners = _assign_tiles_lpt(volumes, 4)
        self.assertEqual(len(owners), len(volumes))
        self.assertTrue(all(0 <= r < 4 for r in owners))

    def test_deterministic(self):
        volumes = [7, 2, 5, 4, 9, 1, 6, 3]
        a = _assign_tiles_lpt(volumes, 4)
        b = _assign_tiles_lpt(list(volumes), 4)
        self.assertEqual(a, b)

    def test_known_lpt_layout(self):
        # Descending: 9,7,6,5,4,3,2,1 greedily to the least-loaded of 4 ranks.
        #  9->r0, 7->r1, 6->r2, 5->r3, 4->r3(5) vs others(9,7,6) -> r3=9,
        #  3->r2(6+3=9), 2->r1(7+2=9), 1->r0(9+1=10). Loads: [10,9,9,9].
        volumes = [9, 7, 6, 5, 4, 3, 2, 1]
        owners = _assign_tiles_lpt(volumes, 4)
        self.assertEqual(self._loads(volumes, owners, 4), [10, 9, 9, 9])

    def test_balance_is_near_optimal(self):
        # Uniform tiles across 8 ranks: perfectly balanced.
        volumes = [10] * 24
        owners = _assign_tiles_lpt(volumes, 8)
        loads = self._loads(volumes, owners, 8)
        self.assertEqual(max(loads), min(loads))

    def test_more_ranks_than_tiles(self):
        volumes = [4, 2, 6]
        owners = _assign_tiles_lpt(volumes, 8)
        # Largest tiles land on distinct least-loaded ranks; no tile is dropped.
        self.assertEqual(sorted(self._loads(volumes, owners, 8), reverse=True)[:3], [6, 4, 2])
        self.assertEqual(len(owners), 3)


if __name__ == "__main__":
    unittest.main()
