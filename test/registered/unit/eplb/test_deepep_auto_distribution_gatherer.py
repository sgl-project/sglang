"""Unit tests for exact EPLB recording with DeepEP auto mode."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepEPAutoSinglePassGatherer(CustomTestCase):
    def setUp(self):
        self.metadata = SimpleNamespace(
            num_layers=2,
            num_physical_experts=8,
            num_local_physical_experts=2,
        )
        with patch.object(expert_distribution, "get_device", return_value="cpu"):
            self.gatherer = expert_distribution._DeepepAutoSinglePassGatherer(
                self.metadata, rank=1
            )

    def test_extend_uses_select_experts_and_ignores_low_latency_hook(self):
        self.gatherer.on_forward_pass_start(SimpleNamespace(is_extend_in_batch=True))
        self.gatherer.on_select_experts(
            layer_idx=0, topk_ids=torch.tensor([[0, 1], [1, -1]])
        )
        self.gatherer.on_deepep_dispatch_low_latency(
            layer_idx=0, local_physical_count_of_layer=torch.tensor([5, 6])
        )

        result = self.gatherer.collect()["global_physical_count"]
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2, 0, 0, 0, 0, 0, 0])))

    def test_decode_uses_rank_local_dispatch_counts_and_ignores_select_hook(self):
        self.gatherer.on_forward_pass_start(SimpleNamespace(is_extend_in_batch=False))
        self.gatherer.on_select_experts(layer_idx=1, topk_ids=torch.tensor([[6, 7]]))
        self.gatherer.on_deepep_dispatch_low_latency(
            layer_idx=1, local_physical_count_of_layer=torch.tensor([3, 4])
        )

        result = self.gatherer.collect()["global_physical_count"]
        self.assertTrue(torch.equal(result[1], torch.tensor([0, 0, 3, 4, 0, 0, 0, 0])))

    def test_decode_excludes_appended_fused_shared_expert_count(self):
        self.gatherer.on_forward_pass_start(SimpleNamespace(is_extend_in_batch=False))
        self.gatherer.on_deepep_dispatch_low_latency(
            layer_idx=0,
            local_physical_count_of_layer=torch.tensor([3, 4, 99]),
        )

        result = self.gatherer.collect()["global_physical_count"]
        self.assertTrue(torch.equal(result[0], torch.tensor([0, 0, 3, 4, 0, 0, 0, 0])))


if __name__ == "__main__":
    import unittest

    unittest.main()
