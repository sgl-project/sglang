import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.kernels.ops.speculative import spec_tree
from sglang.srt.arg_groups.speculative_hook import _check_oot_speculative_algorithm
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_utils, eagle_worker_v2
from sglang.srt.speculative.adaptive_spec_params import adaptive_unsupported_reason
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"
WORKER_MODULE = "sglang.srt.speculative.eagle_worker_v2"
DEVICE_FLAGS = ("_is_cuda", "_is_hip", "_is_musa", "_is_cpu", "_is_npu", "_is_xpu")


class _PlatformWorker:
    pass


def _platform(
    *,
    out_of_tree: bool = True,
    supported: bool = True,
    worker_cls=None,
    graph: bool = True,
) -> Mock:
    platform = Mock(spec=SRTPlatform)
    platform.capabilities = PlatformCapabilities(graph_capture=graph)
    platform.device_name = "custom"
    platform.is_out_of_tree.return_value = out_of_tree
    platform.supports_speculative_algorithm.return_value = supported
    platform.get_speculative_worker_cls.return_value = worker_cls
    return platform


class TestOOTEagleWorkerSelection(CustomTestCase):
    def _create_worker(self, platform: Mock):
        args = ServerArgs(model_path="dummy", device="custom")
        builtin = types.ModuleType(WORKER_MODULE)
        builtin.EAGLEWorkerV2 = sentinel.builtin_worker
        with (
            patch("sglang.srt.platforms.current_platform", platform),
            patch.dict(sys.modules, {WORKER_MODULE: builtin}),
        ):
            return SpeculativeAlgorithm.EAGLE.create_worker(args)

    def test_platform_worker_replaces_builtin(self):
        platform = _platform(worker_cls=sentinel.platform_worker)
        self.assertIs(self._create_worker(platform), sentinel.platform_worker)
        platform.get_speculative_worker_cls.assert_called_once_with(algorithm="EAGLE")

    def test_builtin_worker_when_platform_declines(self):
        declining = _platform(worker_cls=None)
        self.assertIs(self._create_worker(declining), sentinel.builtin_worker)
        in_tree = _platform(out_of_tree=False, worker_cls=sentinel.platform_worker)
        self.assertIs(self._create_worker(in_tree), sentinel.builtin_worker)
        in_tree.get_speculative_worker_cls.assert_not_called()


class TestOOTEagleAlgorithmGate(CustomTestCase):
    def test_unsupported_out_of_tree_platform_is_rejected(self):
        with patch(f"{HOOK_MODULE}.current_platform", _platform(supported=False)):
            with self.assertRaisesRegex(ValueError, "EAGLE speculative decoding"):
                _check_oot_speculative_algorithm(algorithm="EAGLE", device="custom")

    def test_builtin_platforms_are_not_gated(self):
        platform = _platform(out_of_tree=False, supported=False)
        with patch(f"{HOOK_MODULE}.current_platform", platform):
            _check_oot_speculative_algorithm(algorithm="EAGLE", device="cuda")
        platform.supports_speculative_algorithm.assert_not_called()


class TestOOTPlatformWorkerAdaptive(CustomTestCase):
    def _reason(self, platform: Mock):
        args = ServerArgs(
            model_path="dummy", device="custom", speculative_algorithm="EAGLE"
        )
        with patch("sglang.srt.platforms.current_platform", platform):
            return adaptive_unsupported_reason(args)

    def test_platform_worker_disables_adaptive(self):
        reason = self._reason(_platform(worker_cls=_PlatformWorker))
        self.assertIn("_PlatformWorker", reason)

    def test_builtin_worker_keeps_adaptive(self):
        self.assertIsNone(self._reason(_platform(worker_cls=None)))
        self.assertIsNone(
            self._reason(_platform(out_of_tree=False, worker_cls=_PlatformWorker))
        )


class TestOOTEagleKernelDispatch(CustomTestCase):
    """An OOT device has no tree kernel; it must fail loudly, not compute nothing."""

    def _run(self, platform: Mock, fn):
        with contextlib.ExitStack() as stack:
            for flag in DEVICE_FLAGS:
                stack.enter_context(patch.object(spec_tree, flag, False))
            stack.enter_context(patch.object(spec_tree, "current_platform", platform))
            return fn()

    def _build_tree(self):
        return eagle_utils.build_tree_kernel_efficient(
            bonus_tokens=torch.tensor([5]),
            parent_list=torch.zeros((1, 2), dtype=torch.long),
            top_scores_index=torch.zeros((1, 2), dtype=torch.long),
            draft_tokens=torch.tensor([[6, 7]]),
            seq_lens=torch.tensor([4]),
            seq_lens_sum=4,
            topk=1,
            spec_steps=2,
            num_verify_tokens=3,
            tree_mask_mode=spec_tree.TreeMaskMode.QLEN_ONLY,
        )

    def _verify(self):
        return spec_tree.verify_tree_greedy_func(
            predicts=None,
            accept_index=None,
            accept_token_num=None,
            candidates=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            target_predict=None,
        )

    def test_out_of_tree_device_is_rejected(self):
        for name, fn in (("build_tree", self._build_tree), ("verify", self._verify)):
            with self.subTest(op=name):
                with self.assertRaisesRegex(NotImplementedError, "no platform seam"):
                    self._run(_platform(), fn)

    def test_in_tree_verify_fallthrough_is_unchanged(self):
        self.assertEqual(
            self._run(_platform(out_of_tree=False), self._verify), (None, None, None)
        )


class TestOOTSpecGraphRunnerSelection(CustomTestCase):
    def _select(self, platform: Mock, oot_runner_cls):
        def builtin():
            raise KeyError("custom")  # an OOT device is not a key of the maps

        with patch.object(eagle_worker_v2, "current_platform", platform):
            return eagle_worker_v2._select_spec_graph_runner_cls(
                builtin_runner_cls=builtin, oot_runner_cls=oot_runner_cls
            )

    def test_no_capture_without_graph_capture(self):
        oot = Mock(return_value=sentinel.runner)
        self.assertIsNone(self._select(_platform(graph=False), oot))
        oot.assert_not_called()

    def test_builtin_platform_never_consults_the_platform(self):
        oot = Mock(return_value=sentinel.runner)
        with patch.object(
            eagle_worker_v2, "current_platform", _platform(out_of_tree=False)
        ):
            selected = eagle_worker_v2._select_spec_graph_runner_cls(
                builtin_runner_cls=lambda: sentinel.builtin_runner,
                oot_runner_cls=oot,
            )
        self.assertIs(selected, sentinel.builtin_runner)
        oot.assert_not_called()

    def test_out_of_tree_draft_graphs_come_from_the_platform(self):
        """An OOT device name used to index the device-keyed runner maps (KeyError)."""
        platform = _platform()
        runners = {
            phase: Mock(name=phase) for phase in ("draft_decode", "draft_extend")
        }
        platform.get_speculative_graph_runner_cls.side_effect = (
            lambda algorithm, phase: runners[phase]
        )
        worker = SimpleNamespace(
            speculative_algorithm=SpeculativeAlgorithm.EAGLE,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            topk=1,
            target_worker=SimpleNamespace(device="custom"),
            draft_runner=None,
            device="custom",
            gpu_id=0,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            _specialized_graph_memory_usage={},
            _specialized_graph_time_usage={},
        )
        decode_cfg = SimpleNamespace(backend="custom")
        exec_cfg = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode_cfg))
        )
        with contextlib.ExitStack() as stack:
            for flag in DEVICE_FLAGS:
                stack.enter_context(patch.object(eagle_worker_v2, flag, False))
            for name, value in (
                ("current_platform", platform),
                ("check_cuda_graph_backend", lambda *args: False),
                ("get_model", lambda: SimpleNamespace(model_impl="auto")),
                ("get_exec", lambda: exec_cfg),
                ("get_batch_sizes_to_capture", lambda runner: ([1], None)),
                ("get_available_gpu_memory", lambda *args: 0.0),
            ):
                stack.enter_context(patch.object(eagle_worker_v2, name, value))
            eagle_worker_v2.EagleDraftWorker._capture_cuda_graphs(worker)
        for phase, runner in runners.items():
            runner.assert_called_once_with(worker)
        self.assertIs(worker.cuda_graph_runner, runners["draft_decode"].return_value)
        self.assertIs(
            worker.cuda_graph_runner_for_draft_extend,
            runners["draft_extend"].return_value,
        )


if __name__ == "__main__":
    unittest.main()
