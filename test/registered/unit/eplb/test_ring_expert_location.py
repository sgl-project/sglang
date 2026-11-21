import unittest
from types import SimpleNamespace

import torch

from sglang.srt.eplb.expert_location import (
    compute_ring_physical_to_logical_map,
    is_ring_placement,
)
from sglang.srt.layers.muse_spark.muse_eplb_ring import (
    RingLayout,
    copy_ring_slots,
    home_packed,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _metadata(physical_to_logical_map, ep_size):
    return SimpleNamespace(
        physical_to_logical_map_cpu=physical_to_logical_map, ep_size=ep_size
    )


class TestRingExpertLocation(CustomTestCase):
    def test_ring_map(self):
        for ep_size in (2, 4, 8):
            with self.subTest(ep_size=ep_size):
                ring = compute_ring_physical_to_logical_map(
                    num_layers=3, num_logical_experts=256, ep_size=ep_size
                )
                home = 256 // ep_size
                self.assertEqual(tuple(ring.shape), (3, 512))
                self.assertTrue(bool((ring == ring[0]).all()))
                for rank in range(ep_size):
                    slots = ring[0, rank * 2 * home : (rank + 1) * 2 * home].tolist()
                    buddy = (rank + 1) % ep_size
                    self.assertEqual(
                        slots,
                        list(range(rank * home, (rank + 1) * home))
                        + list(range(buddy * home, (buddy + 1) * home)),
                    )
                self.assertTrue(bool((torch.bincount(ring[0]) == 2).all()))

    def test_ring_map_rejects_bad_ep_sizes(self):
        for ep_size in (1, 3):
            with self.subTest(ep_size=ep_size), self.assertRaises(ValueError):
                compute_ring_physical_to_logical_map(
                    num_layers=1, num_logical_experts=256, ep_size=ep_size
                )

    def test_is_ring_placement(self):
        ring = compute_ring_physical_to_logical_map(
            num_layers=4, num_logical_experts=256, ep_size=8
        )
        check = dict(num_layers=4, num_logical_experts=256, ep_size=8)
        self.assertTrue(is_ring_placement(_metadata(ring, 8), **check))
        swapped = ring.clone()
        swapped[0, [0, 1]] = swapped[0, [1, 0]]
        last_layer = ring.clone()
        last_layer[-1, 40] = 7
        buddy_reversed = compute_ring_physical_to_logical_map(
            num_layers=4, num_logical_experts=256, ep_size=8
        )
        for rank in range(8):
            prev = (rank - 1) % 8
            buddy_reversed[:, rank * 64 + 32 : rank * 64 + 64] = torch.arange(
                prev * 32, prev * 32 + 32
            )
        trivial = torch.arange(512).repeat(4, 1) % 256
        for name, metadata in (
            ("none", None),
            ("swapped slots", _metadata(swapped, 8)),
            ("last layer", _metadata(last_layer, 8)),
            ("buddy is the previous rank", _metadata(buddy_reversed, 8)),
            ("trivial with redundant slots", _metadata(trivial, 8)),
            ("missing layer", _metadata(ring[:3], 8)),
            ("ep size", _metadata(ring, 4)),
        ):
            with self.subTest(name):
                self.assertFalse(is_ring_placement(metadata, **check))

    def test_ring_check_under_another_default_device(self):
        # Model construction runs under a CUDA default device; the map must stay on the CPU.
        ring = compute_ring_physical_to_logical_map(
            num_layers=2, num_logical_experts=256, ep_size=8
        )
        with torch.device("meta"):
            self.assertEqual(
                compute_ring_physical_to_logical_map(
                    num_layers=2, num_logical_experts=256, ep_size=8
                ).device.type,
                "cpu",
            )
            self.assertTrue(
                is_ring_placement(
                    _metadata(ring, 8), num_layers=2, num_logical_experts=256, ep_size=8
                )
            )

    def test_copy_ring_slots(self):
        stack = torch.arange(256).view(256, 1, 1).expand(256, 2, 3).contiguous()
        for ep_size, rank in ((8, 0), (8, 3), (8, 7), (2, 1)):
            layout = RingLayout(num_ranks=ep_size, home=256 // ep_size, rank=rank)
            with self.subTest(ep_size=ep_size, rank=rank):
                param = torch.empty(layout.num_slots, 2, 3, dtype=stack.dtype)
                copy_ring_slots(param, stack, layout)
                ring = compute_ring_physical_to_logical_map(
                    num_layers=1, num_logical_experts=256, ep_size=ep_size
                )
                own = ring[
                    0, layout.phys_offset : layout.phys_offset + layout.num_slots
                ]
                self.assertTrue(torch.equal(param[:, 0, 0], own))
        with self.assertRaises(RuntimeError):
            copy_ring_slots(
                torch.empty(32, 2, 3), stack, RingLayout(num_ranks=8, home=32, rank=0)
            )

    def test_home_packed(self):
        layout = RingLayout(num_ranks=8, home=32, rank=0)
        ids = torch.tensor([[0, 31, 32, 255, -1]], dtype=torch.int32)
        packed = (ids << 16) | 0x1234
        out = home_packed(ids, packed, layout)
        self.assertEqual((out[0, :4] >> 16).tolist(), [0, 31, 64, 7 * 64 + 31])
        self.assertTrue(torch.equal(out & 0xFFFF, packed & 0xFFFF))
        self.assertEqual(int(out[0, 4]), int(packed[0, 4]))


if __name__ == "__main__":
    unittest.main()
