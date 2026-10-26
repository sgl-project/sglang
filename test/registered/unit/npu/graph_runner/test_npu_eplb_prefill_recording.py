"""CPU regression tests for NPU prefill recording policy (no device graphs)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu import eplb_statistics as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def make_gatherer():
    gatherer = object.__new__(mod.NpuExpertDistributionGatherer)
    gatherer._expert_location_metadata = SimpleNamespace(num_physical_experts=3)
    gatherer._enable_global_physical_experts = True
    gatherer._data = torch.zeros((1, 3), dtype=torch.int32)
    gatherer._batched_routes = torch.full((1, 8), -1, dtype=torch.int32)
    gatherer._route_offsets = [0]
    gatherer._has_batched_routes = False
    gatherer._batch_collected = False
    gatherer._direct_counting = False
    return gatherer


class TestNpuPrefillRecording(unittest.TestCase):
    def test_direct_counts_then_decode_batching(self):
        gatherer = make_gatherer()
        ids = torch.tensor([[0, 1, 1, -1, 3]], dtype=torch.int32)
        with patch.object(
            mod, "_get_npu_gatherers", return_value={"primary": gatherer}
        ):
            with mod.direct_route_counting():
                gatherer.on_select_experts(0, ids)
                self.assertEqual(gatherer._route_offsets, [0])
                self.assertFalse(gatherer._has_batched_routes)
                self.assertEqual(
                    gatherer.collect()["global_physical_count"].tolist(), [[1, 2, 0]]
                )
            self.assertFalse(gatherer._direct_counting)
            gatherer.reset()
            gatherer.on_select_experts(0, ids)
            self.assertTrue(gatherer._has_batched_routes)
            self.assertEqual(gatherer._data.tolist(), [[0, 0, 0]])
            self.assertEqual(
                gatherer.collect()["global_physical_count"].tolist(), [[1, 2, 0]]
            )

    def test_nested_scope_restores_on_exception(self):
        gatherer = make_gatherer()
        with patch.object(
            mod, "_get_npu_gatherers", return_value={"primary": gatherer}
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with mod.direct_route_counting():
                    with mod.direct_route_counting():
                        self.assertTrue(gatherer._direct_counting)
                    self.assertTrue(gatherer._direct_counting)
                    raise RuntimeError("capture failed")
            self.assertFalse(gatherer._direct_counting)

    def test_disabled_draft_recorder_is_not_modified(self):
        gatherer = make_gatherer()
        recorder = object.__new__(mod.distribution._ExpertDistributionRecorderReal)
        recorder._disable_all = True
        recorder._single_pass_gatherers = {"primary": gatherer}
        with patch.object(
            mod.distribution,
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            with mod.direct_route_counting():
                self.assertFalse(gatherer._direct_counting)


if __name__ == "__main__":
    unittest.main()
