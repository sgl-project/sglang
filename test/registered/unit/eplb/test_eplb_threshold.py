"""EPLB communication and device adaptation retain upstream threshold semantics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEPLBThreshold(unittest.TestCase):
    def accumulator(self, rank=0, threshold=0.9, report_mode="server_log"):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                expert_distribution_recorder_mode="stat",
                expert_distribution_recorder_buffer_size=4,
                expert_balancedness_report_mode=report_mode,
                eplb_min_rebalancing_utilization_threshold=threshold,
            ),
            role="test",
        )
        metadata = SimpleNamespace(num_layers=2, num_physical_experts=8, ep_size=2)
        with (
            patch.object(distribution.torch.distributed, "get_rank", return_value=rank),
            patch.object(
                distribution, "resolve_collector_class", return_value=lambda _: None
            ),
        ):
            return distribution._StatAccumulator(metadata, rank)

    def test_disabled_reporting_keeps_upstream_threshold_behavior(self):
        accumulator = self.accumulator(report_mode="off")
        with patch.object(distribution.torch.distributed, "reduce") as reduce:
            accumulator.append(
                0,
                "primary",
                {"global_physical_count": torch.ones(2, 8, dtype=torch.int32)},
                {},
            )
            self.assertIsNone(accumulator._get_global_average_utilization_rate())
        reduce.assert_not_called()

    def test_threshold_statistics_use_selected_communication_group(self):
        accumulator = self.accumulator()
        group = object()
        with (
            patch.object(distribution, "get_eplb_device_group", return_value=group),
            patch.object(distribution.torch.distributed, "reduce") as reduce,
        ):
            accumulator.append(
                0,
                "primary",
                {"global_physical_count": torch.ones(2, 8, dtype=torch.int32)},
                {},
            )
        self.assertIs(reduce.call_args.kwargs["group"], group)

    def test_threshold_broadcast_uses_runtime_device_on_both_ranks(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                accumulator = self.accumulator(rank)
                if rank == 0:
                    accumulator._history.append(0.25)
                    accumulator._history.append(0.75)

                def broadcast(tensor, src, group=None):
                    self.assertEqual(src, 0)
                    self.assertIsNone(group)
                    self.assertEqual(tensor.device.type, "cpu")
                    self.assertEqual(tensor.dtype, torch.float32)
                    if rank == 0:
                        self.assertEqual(tensor.item(), 0.5)
                    else:
                        tensor.fill_(0.5)

                with patch.object(
                    distribution.torch.distributed, "broadcast", side_effect=broadcast
                ):
                    self.assertEqual(
                        accumulator._get_global_average_utilization_rate(), 0.5
                    )

    def test_default_threshold_keeps_reporting_off_path_inactive(self):
        accumulator = self.accumulator(threshold=1.0, report_mode="off")
        with patch.object(distribution.torch.distributed, "reduce") as reduce:
            accumulator.append(
                0,
                "primary",
                {"global_physical_count": torch.ones(2, 8, dtype=torch.int32)},
                {},
            )
            self.assertIsNone(accumulator._get_global_average_utilization_rate())
        reduce.assert_not_called()

    def test_reporting_still_emits_metrics(self):
        # Isolate metric registration from the process-wide Prometheus registry.
        with patch.object(
            distribution, "resolve_collector_class", return_value=lambda _: None
        ):
            accumulator = self.accumulator(threshold=1.0, report_mode="server_log")
        outputs = {}
        with patch.object(distribution.torch.distributed, "reduce"):
            accumulator.append(
                0,
                "primary",
                {"global_physical_count": torch.ones(2, 8, dtype=torch.int32)},
                outputs,
            )
        self.assertEqual(outputs["metrics"].eplb_balancedness.item(), 1.0)
        self.assertEqual(outputs["metrics"].gpu_physical_count_sum.item(), 16)
        self.assertTrue(outputs["metrics"].reset_server_log_history)


if __name__ == "__main__":
    unittest.main()
