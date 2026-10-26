"""CPU unit tests for the NPU CUDA-graph backend's Python orchestration."""

import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.hardware_backend.npu import eplb_statistics
from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_backend():
    device_module = SimpleNamespace(
        current_device=mock.Mock(return_value=3),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
        Event=mock.Mock(),
    )
    runner = SimpleNamespace(device_module=device_module, enable_torch_compile=False)
    with (
        get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
        mock.patch.object(mod.TorchMemorySaverAdapter, "create", return_value=None),
    ):
        return mod.NPUCudaGraphBackend(runner), runner


class TestNPUCudaGraphBackend(unittest.TestCase):
    def test_capture_one_resets_offsets_and_first_forward_discards_warmups(self):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                expert_distribution_recorder_mode="stat",
                expert_distribution_recorder_buffer_size=4,
            ),
            role="test",
        )
        metadata = SimpleNamespace(
            num_layers=2, num_physical_experts=8, num_local_physical_experts=4
        )
        with mock.patch.object(distribution, "get_device", return_value="cpu"):
            gatherer = eplb_statistics.NpuExpertDistributionGatherer(metadata, 0)
        with mock.patch.object(
            distribution._SinglePassGatherer, "init_new", return_value=gatherer
        ):
            recorder = eplb_statistics.NpuExpertDistributionRecorder(metadata, 0)
        recorder.start_record()
        backend, _ = _make_backend()
        self.addCleanup(backend.cleanup)
        fake_npu = SimpleNamespace(
            NPUGraph=mock.Mock(), graph=lambda *args, **kwargs: nullcontext()
        )
        warmup_hook = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"torch_npu": ModuleType("torch_npu")}),
            mock.patch.object(torch, "npu", fake_npu, create=True),
            mock.patch.object(
                distribution,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            for count in (3, 1):
                calls = []

                def forward():
                    # Missing any warmup/capture reset leaves a nonzero offset.
                    self.assertEqual(gatherer._route_offsets, [0, 0])
                    with recorder.with_current_layer(0):
                        recorder.on_select_experts(torch.full((count,), 2))
                    calls.append(True)
                    return count

                key = ShapeKey(size=count)
                backend.capture_one(key, forward, post_warmup_hook=warmup_hook)
                self.assertEqual(len(calls), 3)
                self.assertTrue(any(backend._eplb_capture_states[key].values()))
                self.assertEqual(backend._outputs[key], count)
        self.assertEqual(warmup_hook.call_count, 4)
        with recorder.with_forward_pass(1, SimpleNamespace()):
            with recorder.with_current_layer(1):
                recorder.on_select_experts(torch.tensor([7]))
        counts = recorder._accumulator._global_physical_count_of_buffered_step.get_all()
        total = counts.sum(0)
        expected = torch.zeros_like(total)
        expected[1, 7] = 1
        torch.testing.assert_close(total, expected)

    def test_replay_update_converts_legacy_sequence_lengths_to_int32_tensor(self):
        backend, runner = _make_backend()
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"
        backend._eplb_capture_states[shape_key] = {"primary": True}

        events = []
        prepare_statistics = mock.Mock(
            side_effect=lambda state: events.append("statistics")
        )
        graph.replay.side_effect = lambda: events.append("replay")
        with mock.patch.object(
            eplb_statistics,
            "on_graph_replay_forward_start",
            prepare_statistics,
        ):
            output = backend.replay_with_input_update(
                shape_key,
                seq_lens=[7, 9],
                attr_name="seq_lens",
                attr_type=torch.empty(0),
            )
        prepare_statistics.assert_called_once_with({"primary": True})
        self.assertEqual(events, ["statistics", "replay"])

        runner.device_module.set_device.assert_called_once_with(3)
        graph.replay.assert_called_once_with()
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(len(update_input), 1)
        self.assertEqual(set(update_input[0]), {"seq_lens"})
        self.assertEqual(update_input[0]["seq_lens"].dtype, torch.int32)
        self.assertTrue(torch.equal(update_input[0]["seq_lens"], torch.tensor([7, 9])))
        self.assertEqual(output, "output")
        backend.cleanup()


if __name__ == "__main__":
    unittest.main()
