"""MegaGate's int32 dispatch maps are built once per algorithm and track EPLB updates."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.mega_gate import _physical_dispatch_maps
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS, NUM_LOGICAL, NUM_DUP = 2, 4, 2


def _metadata(seed: int) -> ExpertLocationMetadata:
    g = torch.Generator().manual_seed(seed)
    num_physical = NUM_LOGICAL * NUM_DUP
    all_physical = torch.full((NUM_LAYERS, NUM_LOGICAL, num_physical), -1)
    for layer in range(NUM_LAYERS):
        all_physical[layer, :, :NUM_DUP] = torch.randperm(
            num_physical, generator=g
        ).view(NUM_LOGICAL, NUM_DUP)
    num_valid = torch.randint(
        1, NUM_DUP + 1, (NUM_LAYERS, NUM_LOGICAL), generator=g, dtype=torch.int64
    )
    physical_to_logical = torch.zeros((NUM_LAYERS, num_physical), dtype=torch.int64)
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical,
        physical_to_logical_map_cpu=physical_to_logical.clone(),
        logical_to_all_physical_map=all_physical,
        logical_to_all_physical_map_cpu=all_physical.clone(),
        logical_to_all_physical_map_num_valid=num_valid,
        ep_size=1,
        logical_to_rank_dispatch_physical_map=all_physical[:, :, 0].clone(),
    )


def _info(metadata, layer, algorithm, with_maps=True):
    return ExpertLocationDispatchInfo(
        ep_dispatch_algorithm=algorithm,
        partial_logical_to_rank_dispatch_physical_map=metadata.logical_to_rank_dispatch_physical_map[
            layer
        ],
        partial_logical_to_all_physical_map=metadata.logical_to_all_physical_map[layer],
        partial_logical_to_all_physical_map_num_valid=metadata.logical_to_all_physical_map_num_valid[
            layer
        ],
        num_physical_experts=metadata.num_physical_experts,
        mega_gate_maps=(
            tuple(m[layer] for m in metadata.mega_gate_maps(algorithm))
            if with_maps
            else None
        ),
    )


class TestMegaGateDispatchMaps(CustomTestCase):
    def test_built_once_per_algorithm_on_first_use(self):
        metadata = _metadata(0)
        self.assertEqual(metadata._mega_gate_maps, {})
        maps = metadata.mega_gate_maps("dynamic")
        self.assertIs(metadata.mega_gate_maps("dynamic"), maps)
        self.assertEqual(list(metadata._mega_gate_maps), ["dynamic"])

    def test_gate_reads_cached_maps_and_update_refreshes_them(self):
        metadata, new = _metadata(0), _metadata(1)
        dynamic, static = _info(metadata, 1, "dynamic"), _info(metadata, 1, "static")
        phys, count = _physical_dispatch_maps(dynamic)
        rank_map, ones = _physical_dispatch_maps(static)
        self.assertIs(phys, dynamic.mega_gate_maps[0])
        self.assertIs(rank_map, static.mega_gate_maps[0])
        self.assertEqual((phys.dtype, count.dtype), (torch.int32, torch.int32))
        self.assertTrue(torch.equal(ones, torch.ones(NUM_LOGICAL, dtype=torch.int32)))

        metadata.update(new, update_layer_ids=[1])
        self.assertTrue(torch.equal(phys.long(), new.logical_to_all_physical_map[1]))
        self.assertTrue(
            torch.equal(count.long(), new.logical_to_all_physical_map_num_valid[1])
        )
        self.assertTrue(
            torch.equal(
                rank_map[:, 0].long(), new.logical_to_rank_dispatch_physical_map[1]
            )
        )

    def test_without_cached_maps_the_gate_casts(self):
        metadata = _metadata(0)
        phys, count = _physical_dispatch_maps(_info(metadata, 0, "dynamic", False))
        self.assertEqual(metadata._mega_gate_maps, {})
        self.assertEqual((phys.dtype, count.dtype), (torch.int32, torch.int32))
        self.assertTrue(
            torch.equal(phys.long(), metadata.logical_to_all_physical_map[0])
        )


if __name__ == "__main__":
    unittest.main()
