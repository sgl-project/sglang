"""Independent CPU histogram oracles for the NPU counter capture lifecycle."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.hardware_backend.npu.eplb_statistics import (
    NpuExpertDistributionGatherer,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
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
            return NpuExpertDistributionGatherer(self.metadata, 0)

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

    def test_overflow_only_pass_skips_batch_work(self):
        gatherer = self.gatherer(capacity=8)
        calls = [(layer, torch.arange(18) % 8) for layer in range(2)]
        with (
            patch.object(
                gatherer._data, "scatter_add_", wraps=gatherer._data.scatter_add_
            ) as batch,
            patch.object(
                gatherer._batched_routes, "fill_", wraps=gatherer._batched_routes.fill_
            ) as clear,
        ):
            for _ in range(2):
                gatherer.reset()
                self.check_counts(gatherer, calls)
            batch.assert_not_called()
            clear.assert_not_called()

    def test_graph_and_eager_shapes_restore_capture_state(self):
        gatherer = self.gatherer(capacity=8)
        recorder = distribution._ExpertDistributionRecorderReal.__new__(
            distribution._ExpertDistributionRecorderReal
        )
        recorder._disable_all = False
        recorder._single_pass_gatherers = {"primary": gatherer}
        backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
        backend._graphs, backend._outputs, backend._eplb_capture_states = {}, {}, {}
        calls = {
            "small": [(0, torch.tensor([1, 2, 2]))],
            "large": [(1, torch.arange(18) % 8)],
        }
        for key, routes in calls.items():
            recorder.on_graph_capture_forward_start()
            for layer, ids in routes:
                gatherer.on_select_experts(layer, ids)
            backend._eplb_capture_states[key] = recorder.get_graph_capture_state()

            def replay(routes=routes):
                # Emulate captured tensor writes without executing Python hooks.
                for layer, ids in routes:
                    if ids.numel() <= 8:
                        gatherer._batched_routes[layer, : ids.numel()].copy_(ids)
                    else:
                        gatherer._data[layer].copy_(self.oracle([(layer, ids)])[layer])

            backend._graphs[key] = SimpleNamespace(replay=replay)
            backend._outputs[key] = key
        self.assertEqual(
            backend._eplb_capture_states,
            {"small": {"primary": True}, "large": {"primary": False}},
        )
        with patch.object(
            distribution,
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            for key in ("large", "small", "large", "small", "small"):
                gatherer.reset()
                backend.replay(key, None)
                torch.testing.assert_close(
                    gatherer.collect()["global_physical_count"], self.oracle(calls[key])
                )
                # The next eager overflow must not reuse captured routes.
                gatherer.reset()
                self.check_counts(gatherer, calls["large"])
            gatherer.reset()
            gatherer.on_select_experts(0, torch.tensor([7]))
            backend.replay("large", None)
            torch.testing.assert_close(
                gatherer.collect()["global_physical_count"],
                self.oracle([(0, torch.tensor([7])), *calls["large"]]),
            )
            gatherer.reset()
            with recorder.disable_this_region():
                self.assertEqual(recorder.get_graph_capture_state(), {})
                recorder.on_graph_replay_forward_start({"primary": True})
            self.assertFalse(gatherer._has_batched_routes)

    def test_generated_sequences_match_independent_histogram(self):
        generator = torch.Generator().manual_seed(910)
        for capacity in (8, 96):
            gatherer = self.gatherer(capacity)
            for _ in range(30):
                gatherer.reset()
                calls = [
                    (
                        int(torch.randint(2, (), generator=generator)),
                        torch.randint(
                            -2,
                            10,
                            (int(torch.randint(130, (), generator=generator)),),
                            generator=generator,
                        ),
                    )
                    for _ in range(8)
                ]
                self.check_counts(gatherer, calls)

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
