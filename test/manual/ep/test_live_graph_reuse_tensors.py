"""Tensor/graph regressions for live growth; run in the serving CUDA image."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.eplb import expert_location as location


class TestLiveReuseTensors(unittest.TestCase):
    def make_gatherer(self, device, max_ep_size=4):
        metadata = SimpleNamespace(
            num_layers=1,
            num_physical_experts=4,
            num_local_physical_experts=2,
            ep_size=2,
        )
        with (
            patch.object(distribution, "get_device", return_value=device),
            patch.object(
                distribution,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_reuse_cuda_graph=True)
                ),
            ),
            patch.object(
                distribution,
                "get_parallel",
                return_value=SimpleNamespace(max_ep_size=max_ep_size),
            ),
        ):
            gatherer = distribution._SelectExpertsSinglePassGatherer(metadata, 0)
        return metadata, gatherer

    def test_shrink_only_without_explicit_capacity_keeps_launch_width(self):
        _, gatherer = self.make_gatherer("cpu", max_ep_size=None)
        self.assertEqual(tuple(gatherer._data.shape), (1, 4))

    def test_counter_grows_and_shrinks_without_reallocation(self):
        metadata, gatherer = self.make_gatherer("cpu")
        pointer = gatherer._data.data_ptr()
        self.assertEqual(tuple(gatherer._data.shape), (1, 8))
        metadata.num_physical_experts = 8
        gatherer.on_select_experts(0, torch.tensor([0, 4, 7, -1, 8]))
        self.assertEqual(
            gatherer.collect()["global_physical_count"].tolist(),
            [[1, 0, 0, 0, 1, 0, 0, 1]],
        )
        metadata.num_physical_experts = 4
        self.assertEqual(
            tuple(gatherer.collect()["global_physical_count"].shape), (1, 4)
        )
        self.assertEqual(pointer, gatherer._data.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_captured_counter_accepts_new_experts_after_growth(self):
        metadata, gatherer = self.make_gatherer("cuda")
        topk = torch.tensor([0, 1, -1], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                gatherer.on_select_experts(0, topk)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gatherer.on_select_experts(0, topk)
        metadata.num_physical_experts = 8
        topk.copy_(torch.tensor([4, 7, -1], device="cuda"))
        gatherer.reset()
        graph.replay()
        self.assertEqual(
            gatherer.collect()["global_physical_count"].tolist(),
            [[0, 0, 0, 0, 1, 0, 0, 1]],
        )

    def test_nonzero_rank_broadcast_preserves_graph_routing_storage(self):
        def metadata(width, value):
            return location.ExpertLocationMetadata(
                physical_to_logical_map=torch.arange(width).reshape(1, width) % 4,
                physical_to_logical_map_cpu=torch.arange(width).reshape(1, width) % 4,
                logical_to_all_physical_map=torch.full((1, 4, 8), value),
                logical_to_all_physical_map_cpu=torch.full((1, 4, 8), value),
                logical_to_all_physical_map_num_valid=torch.full((1, 4), value),
                logical_to_rank_dispatch_physical_map=torch.full((1, 4), value),
                ep_size=width // 2,
            )

        original = metadata(8, 0)
        updated = metadata(8, 1)
        captured = [
            original.logical_to_all_physical_map,
            original.logical_to_all_physical_map_num_valid,
            original.logical_to_rank_dispatch_physical_map,
        ]
        pointers = [t.data_ptr() for t in captured]
        with (
            patch.object(
                location, "get_global_expert_location_metadata", return_value=original
            ),
            patch.object(
                location,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_reuse_cuda_graph=True)
                ),
            ),
            patch.object(
                location.ExpertLocationMetadata, "init_by_mapping", return_value=updated
            ),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=4),
            patch(
                "sglang.srt.elastic_ep.elastic_ep.share_expert_map_via_store",
                return_value=True,
            ),
        ):
            actual = location.broadcast_global_expert_location_metadata(
                None, moe_ep_rank=1
            )
        self.assertIs(actual, original)
        self.assertEqual(pointers, [t.data_ptr() for t in captured])
        self.assertTrue(all(torch.all(t == 1).item() for t in captured))


if __name__ == "__main__":
    unittest.main()
