"""A scale adopts its layout in place, and MegaGate's cached int32 maps must follow:
they are copies, not aliases, and graphs replay them."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.test.test_utils import CustomTestCase

LAYERS, LOGICAL, DUP = 2, 4, 2


def _metadata(offset: int, ep_size: int) -> ExpertLocationMetadata:
    meta = ExpertLocationMetadata.__new__(ExpertLocationMetadata)
    meta._mega_gate_maps = {}
    all_physical = torch.arange(LAYERS * LOGICAL * DUP).view(LAYERS, LOGICAL, DUP)
    meta.logical_to_all_physical_map = all_physical + offset
    meta.logical_to_all_physical_map_cpu = meta.logical_to_all_physical_map.clone()
    meta.logical_to_all_physical_map_num_valid = torch.full(
        (LAYERS, LOGICAL), 1 + offset % 2, dtype=torch.int64
    )
    meta.logical_to_rank_dispatch_physical_map = (
        torch.arange(LAYERS * LOGICAL).view(LAYERS, LOGICAL) + offset
    )
    meta.physical_to_logical_map = torch.zeros(LAYERS, ep_size, dtype=torch.int64)
    meta.physical_to_logical_map_cpu = meta.physical_to_logical_map.clone()
    meta.ep_size = ep_size
    return meta


class TestAdoptScaledMegaGate(CustomTestCase):
    def test_cached_maps_follow_the_adopted_layout_in_place(self):
        current = _metadata(0, ep_size=8)
        static = current.mega_gate_maps("static")
        dynamic = current.mega_gate_maps("dynamic")

        current.adopt_scaled_in_place(_metadata(101, ep_size=4))

        self.assertIs(current.mega_gate_maps("static")[0], static[0])
        self.assertIs(current.mega_gate_maps("dynamic")[0], dynamic[0])
        self.assertTrue(
            torch.equal(
                static[0][..., 0],
                current.logical_to_rank_dispatch_physical_map.to(torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(dynamic[0], current.logical_to_all_physical_map.to(torch.int32))
        )
        self.assertTrue(
            torch.equal(
                dynamic[1],
                current.logical_to_all_physical_map_num_valid.to(torch.int32),
            )
        )
        self.assertEqual(int(dynamic[1][0, 0]), 2)


if __name__ == "__main__":
    unittest.main()
