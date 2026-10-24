"""Ascend sender statistics across DeepEP modes; CPU-only count oracles."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNpuExpertDistribution(unittest.TestCase):
    def setUp(self):
        self.metadata = SimpleNamespace(
            num_layers=2, num_physical_experts=8, num_local_physical_experts=4
        )
        self.configure()
        for item in (
            patch.object(distribution, "get_device", return_value="cpu"),
            patch.object(distribution, "_use_npu_sender_statistics", return_value=True),
        ):
            item.start()
            self.addCleanup(item.stop)

    def configure(self, mode="auto", enabled=True):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                moe_a2a_backend="deepep",
                deepep_mode=mode,
                expert_distribution_recorder_mode="stat" if enabled else None,
                expert_distribution_recorder_buffer_size=4,
            ),
            role="test",
        )

    def gatherer(self):
        return distribution._SinglePassGatherer.init_new(self.metadata, rank=1)

    def recorder(self):
        result = distribution.ExpertDistributionRecorder.init_new(self.metadata, rank=1)
        result.start_record()
        return result

    def test_all_npu_modes_use_sender_counts(self):
        for mode in ("normal", "low_latency", "auto"):
            with self.subTest(mode=mode):
                self.configure(mode)
                self.assertIs(
                    type(self.gatherer()),
                    distribution._NpuBatchedSelectExpertsSinglePassGatherer,
                )

    def test_padding_and_invalid_routes_are_ignored(self):
        gatherer = self.gatherer()
        gatherer.on_select_experts(0, torch.tensor([[0, 7, -1], [0, 2, 8], [-2, 6, 2]]))
        torch.testing.assert_close(
            gatherer.collect()["global_physical_count"],
            torch.tensor([[2, 0, 2, 0, 0, 0, 1, 1], [0] * 8], dtype=torch.int32),
        )

    def test_receiver_hook_does_not_double_count(self):
        gatherer = self.gatherer()
        gatherer.on_select_experts(0, torch.tensor([[4, 4, 0]]))
        gatherer.on_deepep_dispatch_low_latency(0, torch.tensor([99, 99, 99, 99]))
        expected = torch.tensor([[1, 0, 0, 0, 2, 0, 0, 0], [0] * 8], dtype=torch.int32)
        for _ in range(2):
            torch.testing.assert_close(
                gatherer.collect()["global_physical_count"], expected
            )

    def test_idle_rank_and_reset_preserve_buffer(self):
        gatherer = self.gatherer()
        pointer = gatherer._data.data_ptr()
        gatherer.on_select_experts(1, torch.empty(0, 2, dtype=torch.int32))
        self.assertEqual(torch.count_nonzero(gatherer._data).item(), 0)
        gatherer.on_select_experts(1, torch.tensor([[7, 7]]))
        gatherer.reset()
        self.assertEqual(pointer, gatherer._data.data_ptr())
        self.assertEqual(torch.count_nonzero(gatherer._data).item(), 0)

    def test_window_evicts_old_passes(self):
        recorder = self.recorder()
        for step, count in enumerate((100, 200, 1, 2, 3, 4)):
            with recorder.with_forward_pass(step, SimpleNamespace()):
                with recorder.with_current_layer(0):
                    recorder.on_select_experts(torch.full((count, 1), 0))
        counts = recorder._accumulator._global_physical_count_of_buffered_step.get_all()
        torch.testing.assert_close(
            counts.sum(0), torch.tensor([[10, 0, 0, 0, 0, 0, 0, 0], [0] * 8])
        )

    def test_disabled_eplb_uses_noop_recorder(self):
        self.configure(enabled=False)
        recorder = distribution.ExpertDistributionRecorder.init_new(
            self.metadata, rank=1
        )
        self.assertIsInstance(recorder, distribution._ExpertDistributionRecorderNoop)
        self.assertFalse(recorder.recording)
        recorder.on_select_experts(torch.tensor([[0, 1]]))

    def test_capture_counts_cleared_on_first_real_pass(self):
        recorder = distribution.ExpertDistributionRecorder.init_new(
            self.metadata, rank=1
        )
        gatherer = next(iter(recorder._single_pass_gatherers.values()))
        pointer = gatherer._data.data_ptr()
        capture = SimpleNamespace(is_current_stream_capturing=lambda: True)
        with patch.object(torch, "get_device_module", return_value=capture):
            with recorder.with_current_layer(0):
                recorder.on_select_experts(torch.tensor([[7, 7, 7]]))
        recorder.start_record()
        with recorder.with_forward_pass(0, SimpleNamespace()):
            with recorder.with_current_layer(0):
                recorder.on_select_experts(torch.tensor([[0, 1]]))
        self.assertEqual(pointer, gatherer._data.data_ptr())
        counts = recorder._accumulator._global_physical_count_of_buffered_step.get_all()
        torch.testing.assert_close(
            counts.sum(0), torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [0] * 8])
        )

    def test_other_devices_keep_existing_factory_behavior(self):
        with patch.object(
            distribution, "_use_npu_sender_statistics", return_value=False
        ):
            for mode, cls in (
                ("normal", distribution._SelectExpertsSinglePassGatherer),
                ("low_latency", distribution._DeepepLowLatencySinglePassGatherer),
            ):
                self.configure(mode)
                self.assertIs(type(self.gatherer()), cls)
            self.configure("auto")
            with self.assertRaises(NotImplementedError):
                self.gatherer()


if __name__ == "__main__":
    unittest.main()
