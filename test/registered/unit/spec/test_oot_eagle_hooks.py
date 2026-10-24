import contextlib
import sys
import types
import unittest
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.kernels.ops.speculative import spec_tree
from sglang.srt.arg_groups.speculative_hook import _check_oot_speculative_algorithm
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_utils
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
    *, out_of_tree: bool = True, supported: bool = True, worker_cls=None
) -> Mock:
    platform = Mock(spec=SRTPlatform)
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


if __name__ == "__main__":
    unittest.main()
