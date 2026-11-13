"""CPU contracts for optional NPU pinned-host allocator initialization."""

import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu import pinned_host
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPinnedHostWarmup(CustomTestCase):
    def test_gates_avoid_allocating_on_disabled_platforms_and_drafts(self):
        for enabled, device, draft in (
            (False, "npu", False),
            (True, "cpu", False),
            (True, "cuda", False),
            (True, "npu", True),
        ):
            with (
                self.subTest(enabled=enabled, device=device, draft=draft),
                envs.SGLANG_NPU_PREWARM_PINNED_HOST.override(enabled),
                patch.object(pinned_host.torch, "empty") as allocate,
            ):
                pinned_host.maybe_prewarm_pinned_host(
                    device=device, is_draft_worker=draft
                )
                allocate.assert_not_called()

    def test_requires_explicit_positive_reserve_before_allocating(self):
        for value in (
            "",
            "expandable_segments:False",
            "pinned_reserve_segment_size_mb:0",
            "pinned_reserve_segment_size_mb:-1",
            "pinned_reserve_segment_size_mb:abc",
            "pinned_reserve_segment_size_mb:64,pinned_reserve_segment_size_mb:128",
        ):
            with (
                self.subTest(config=value),
                envs.SGLANG_NPU_PREWARM_PINNED_HOST.override(True),
                patch.dict(os.environ, {"PYTORCH_NPU_ALLOC_CONF": value}),
                patch.object(pinned_host.torch, "empty") as allocate,
            ):
                with self.assertRaisesRegex(
                    ValueError, "positive.*pinned_reserve_segment_size_mb"
                ):
                    pinned_host.maybe_prewarm_pinned_host(
                        device="npu", is_draft_worker=False
                    )
                allocate.assert_not_called()

    def test_initializes_native_allocator_and_releases_temporary_tensor(self):
        class HostTensor:
            def is_pinned(self):
                return True

        allocations = []

        def allocate(*args, **kwargs):
            tensor = HostTensor()
            allocations.append(weakref.ref(tensor))
            self.assertEqual(args, (1,))
            self.assertEqual(
                kwargs,
                {"dtype": pinned_host.torch.uint8, "device": "cpu", "pin_memory": True},
            )
            return tensor

        for config in (
            "expandable_segments:False,pinned_reserve_segment_size_mb:64",
            " pinned_reserve_segment_size_mb : 128 ,expandable_segments:False",
        ):
            with (
                self.subTest(config=config),
                envs.SGLANG_NPU_PREWARM_PINNED_HOST.override(True),
                patch.dict(os.environ, {"PYTORCH_NPU_ALLOC_CONF": config}),
                patch.object(pinned_host.torch, "empty", side_effect=allocate),
            ):
                pinned_host.maybe_prewarm_pinned_host(
                    device="npu", is_draft_worker=False
                )
                self.assertIsNone(allocations[-1]())
                self.assertEqual(os.environ["PYTORCH_NPU_ALLOC_CONF"], config)

    def test_native_error_and_unpinned_result_fail_startup(self):
        with (
            envs.SGLANG_NPU_PREWARM_PINNED_HOST.override(True),
            patch.dict(
                os.environ,
                {"PYTORCH_NPU_ALLOC_CONF": "pinned_reserve_segment_size_mb:64"},
            ),
        ):
            with patch.object(
                pinned_host.torch,
                "empty",
                side_effect=RuntimeError("allocation failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                    pinned_host.maybe_prewarm_pinned_host(
                        device="npu", is_draft_worker=False
                    )
            with patch.object(
                pinned_host.torch,
                "empty",
                return_value=SimpleNamespace(is_pinned=lambda: False),
            ):
                with self.assertRaisesRegex(RuntimeError, "unpinned"):
                    pinned_host.maybe_prewarm_pinned_host(
                        device="npu", is_draft_worker=False
                    )

    def test_model_runner_initializes_host_allocator_after_loading(self):
        maybe_stub_sgl_kernel()
        from sglang.srt.model_executor import model_runner

        class StopAfterWarmup(Exception):
            pass

        events = []
        runner = Mock()
        runner.device = "npu"
        runner.is_draft_worker = False
        runner.model_config = SimpleNamespace(is_layer_types_hybrid_swa=False)
        runner.load_model.side_effect = lambda: events.append("weights")

        def warm(**kwargs):
            self.assertEqual(kwargs, {"device": "npu", "is_draft_worker": False})
            events.append("pinned")

        def topk(**kwargs):
            events.append("topk")
            raise StopAfterWarmup

        with (
            patch.object(model_runner, "ExpertLocationUpdater"),
            patch.object(model_runner, "create_sampler"),
            patch.object(pinned_host, "maybe_prewarm_pinned_host", side_effect=warm),
            patch.object(model_runner, "prepare_moe_topk", side_effect=topk),
        ):
            with self.assertRaises(StopAfterWarmup):
                model_runner.ModelRunner.initialize(runner)
        self.assertEqual(events, ["weights", "pinned", "topk"])


if __name__ == "__main__":
    unittest.main()
