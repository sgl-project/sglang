"""Target EPLB statistics must exclude draft forwards and graph capture."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.hardware_backend.npu import eplb as npu_eplb
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftDistributionIsolation(unittest.TestCase):
    def setUp(self):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                moe_a2a_backend="deepep",
                deepep_mode="low_latency",
                expert_distribution_recorder_mode="stat",
                expert_distribution_recorder_buffer_size=4,
            ),
            role="test",
        )
        device_patch = patch.object(distribution, "get_device", return_value="cpu")
        device_patch.start()
        self.addCleanup(device_patch.stop)
        npu_policy_patch = patch.object(
            npu_eplb, "get_device", return_value=SimpleNamespace(device="npu")
        )
        npu_policy_patch.start()
        self.addCleanup(npu_policy_patch.stop)
        self.recorder = distribution.ExpertDistributionRecorder.init_new(
            SimpleNamespace(
                num_layers=2, num_physical_experts=8, num_local_physical_experts=4
            ),
            rank=0,
        )
        distribution.set_global_expert_distribution_recorder(self.recorder)
        self.recorder.start_record()

    def count(self, value):
        self.recorder.on_deepep_dispatch_low_latency(torch.tensor([value, 0, 0, 0]))

    def total(self):
        return self.recorder._accumulator._global_physical_count_of_buffered_step.get_all().sum(
            dim=0
        )

    def test_nested_disabled_forward_preserves_target_pass(self):
        with self.recorder.with_forward_pass(1, SimpleNamespace()):
            with self.recorder.with_current_layer(0):
                self.count(3)
                with self.recorder.disable_this_region():
                    with self.recorder.with_forward_pass(100, SimpleNamespace()):
                        self.count(99)
                self.count(1)
        torch.testing.assert_close(
            self.total(), torch.tensor([[4, 0, 0, 0, 0, 0, 0, 0], [0] * 8])
        )

    def test_exception_restores_target_recording(self):
        with self.assertRaisesRegex(RuntimeError, "draft failed"):
            with self.recorder.disable_this_region():
                with self.recorder.with_forward_pass(100, SimpleNamespace()):
                    self.count(99)
                    raise RuntimeError("draft failed")
        with self.recorder.with_forward_pass(1, SimpleNamespace()):
            with self.recorder.with_current_layer(0):
                self.count(2)
        self.assertEqual(self.total().sum().item(), 2)

    def runner(self, is_draft):
        from sglang.srt.model_executor import model_runner as module

        runner = SimpleNamespace(
            is_draft_worker=is_draft,
            forward_pass_id=0,
            msprobe_debugger=None,
            canary_manager=None,
            enable_elastic_ep=False,
            eplb_manager=None,
            graph_memory_usage=None,
            graph_time_usage=None,
            eager_runner=None,
        )
        return module, runner

    def test_actual_runner_forward_excludes_draft_pass(self):
        module, target = self.runner(False)
        _, draft = self.runner(True)

        def target_forward(*args):
            with self.recorder.with_current_layer(0):
                self.count(2)
            return SimpleNamespace()

        def draft_forward(*args):
            # DSpark's MoE callback has no target layer scope. A None index
            # would broadcast these counts across all target layers.
            self.count(99)
            return SimpleNamespace()

        target._forward_raw = target_forward
        draft._forward_raw = draft_forward
        batch = SimpleNamespace(apply_deprecated_skip_attn_backend_init=lambda _: None)
        with (
            patch.object(module, "build_step_span_name", return_value="test"),
            patch.object(
                module, "profile_range", return_value=contextlib.nullcontext()
            ),
            patch.object(module, "get_global_experts_capturer", return_value=None),
            patch.object(module, "get_global_indexer_capturer", return_value=None),
        ):
            module.ModelRunner.forward(target, batch)
            module.ModelRunner.forward(draft, batch)
            module.ModelRunner.forward(target, batch)
        torch.testing.assert_close(
            self.total(), torch.tensor([[4, 0, 0, 0, 0, 0, 0, 0], [0] * 8])
        )

    def test_draft_capture_entrypoints_do_not_record_target_hooks(self):
        module, runner = self.runner(True)
        runner.model = None
        capture = SimpleNamespace(
            eager_runner=None,
            prefill=SimpleNamespace(runner=None),
            decode=SimpleNamespace(runner=None),
            runner=None,
            memory_usage=None,
            time_usage=None,
        )

        def capture_hook(*args, **kwargs):
            self.count(99)
            return capture

        with (
            patch.object(module, "prebuild_deepep_v2_buffer"),
            patch.object(module, "capture_cuda_graphs", side_effect=capture_hook),
            patch.object(module, "capture_decode_graph", side_effect=capture_hook),
            patch.object(module, "capture_prefill_graph", side_effect=capture_hook),
            patch.object(module, "replace_graph_memory_usage", return_value=None),
            patch.object(module, "replace_graph_time_usage", return_value=None),
            patch.object(
                torch,
                "get_device_module",
                return_value=SimpleNamespace(is_current_stream_capturing=lambda: True),
            ),
        ):
            for recording in (True, False):
                with self.subTest(recording=recording):
                    if not recording:
                        self.recorder.stop_record()
                    module.ModelRunner.init_cuda_graphs(runner)
                    module.ModelRunner.init_decode_cuda_graph(runner)
                    module.ModelRunner.init_prefill_cuda_graph(
                        runner, force_for_draft_worker=True
                    )
        for gatherer in self.recorder._single_pass_gatherers.values():
            self.assertEqual(gatherer._data.sum().item(), 0)
        # This checks the Python capture-time hook boundary, not NPU graphs.
        self.assertEqual(self.total().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
