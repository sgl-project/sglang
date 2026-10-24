"""Independent CPU histogram oracles for the NPU counter capture lifecycle."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuBatchedStatistics(unittest.TestCase):
    def setUp(self):
        self.metadata = SimpleNamespace(
            num_layers=2, num_physical_experts=8, num_local_physical_experts=4
        )
        item = patch.object(distribution, "get_device", return_value="cpu")
        item.start()
        self.addCleanup(item.stop)

    def gatherer(self, capacity=96):
        with patch.dict(
            os.environ, {"SGLANG_NPU_EPLB_STATS_MAX_ROUTES": str(capacity)}
        ):
            return distribution._NpuBatchedSelectExpertsSinglePassGatherer(
                self.metadata, 0
            )

    def oracle(self, calls):
        expected = torch.zeros(2, 8, dtype=torch.int32)
        for layer, values in calls:
            ids = values.flatten().long()
            expected[layer] += torch.bincount(
                ids[(ids >= 0) & (ids < 8)], minlength=8
            ).int()
        return expected

    def check_counts(self, gatherer, calls):
        for layer, values in calls:
            gatherer.on_select_experts(layer, values)
        expected = self.oracle(calls)
        for _ in range(2):
            torch.testing.assert_close(
                gatherer.collect()["global_physical_count"], expected
            )

    def test_padding_empty_layers_and_invalid_routes(self):
        self.check_counts(
            self.gatherer(),
            [(0, torch.tensor([[0, 7, -1], [8, -2, 0]], dtype=torch.int32))],
        )

    def test_repeated_layer_overflow_retains_all_routes(self):
        self.check_counts(
            self.gatherer(capacity=8),
            [(0, torch.tensor([[value, value, -1]])) for value in (0, 1, 7)],
        )

    def test_large_eager_pass_uses_exact_fallback(self):
        self.check_counts(
            self.gatherer(capacity=8),
            [(1, torch.arange(-1, 12).reshape(1, -1))],
        )

    def test_capture_boundaries_do_not_accumulate_python_offsets(self):
        gatherer = self.gatherer()
        data_pointer = gatherer._data.data_ptr()
        route_pointer = gatherer._batched_routes.data_ptr()
        for tokens in (16, 8, 1):
            for _ in range(3):
                # Capture warmups bypass the normal recorder reset.
                gatherer.on_graph_capture_forward_start()
                for layer in range(2):
                    gatherer.on_select_experts(layer, torch.full((tokens, 6), layer))
                self.assertEqual(gatherer._route_offsets, [tokens * 6] * 2)
        gatherer.reset()
        self.check_counts(gatherer, [(0, torch.tensor([[2, 2, 7]]))])
        self.assertEqual(gatherer._data.data_ptr(), data_pointer)
        self.assertEqual(gatherer._batched_routes.data_ptr(), route_pointer)

    def test_capacity_validation(self):
        for capacity in (0, 65537):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.gatherer(capacity)

    def test_noop_capture_callback_is_safe(self):
        distribution._ExpertDistributionRecorderNoop().on_graph_capture_forward_start()


if __name__ == "__main__":
    unittest.main()
