"""CPU tests for the DeepGEMM MegaMoE locality lifecycle."""

import math
import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMegaMoeLocality(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(
                mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE_WEIGHTS,
                "get",
                return_value=True,
            )
        )
        stack.enter_context(patch.object(mega_moe, "_device_sm", 100))
        stack.enter_context(
            patch.object(
                mega_moe,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_megamoe=lambda: True),
            )
        )
        self.exec_context = SimpleNamespace(moe=SimpleNamespace(enable_eplb=False))
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_exec", return_value=self.exec_context)
        )
        self.synchronize = stack.enter_context(patch.object(torch.cuda, "synchronize"))

        def localize(tensor):
            logical = tensor.unflatten(-2, (2, -1)).movedim(-3, 0)
            strides = torch.empty(logical.shape).stride()
            domain_stride = math.prod(logical.shape[1:]) + 16
            result = torch.empty_strided(
                logical.shape,
                (domain_stride, *strides[1:]),
                dtype=tensor.dtype,
            )
            return result.copy_(logical)

        self.deep_gemm = ModuleType("deep_gemm")
        self.deep_gemm.locality_domain = SimpleNamespace(
            is_localization_available=MagicMock(return_value=True)
        )
        self.deep_gemm.localize = MagicMock(side_effect=localize)
        self.deep_gemm.destroy_localizer = MagicMock()
        stack.enter_context(patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}))

        w13 = torch.nn.Parameter(
            torch.arange(64, dtype=torch.int8).reshape(2, 8, 4), requires_grad=False
        )
        w2 = torch.nn.Parameter(
            torch.arange(32, dtype=torch.int8).reshape(2, 4, 4), requires_grad=False
        )
        self.experts = SimpleNamespace(
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
            _mega_moe_w4a4=False,
            w13_weight=w13,
            w2_weight=w2,
            mega_l1_weights=(w13.data, torch.ones(2, 8)),
            mega_l2_weights=(w2.data, torch.ones(2, 4)),
        )
        self.moe = SimpleNamespace(
            experts=self.experts,
            mega_shared_l1_weights=(
                torch.arange(32, dtype=torch.int8).reshape(8, 4),
                torch.ones(8),
            ),
            mega_shared_l2_weights=(
                torch.arange(16, dtype=torch.int8).reshape(4, 4),
                torch.ones(4),
            ),
        )
        # A module alias must not localize the same expert parameters twice.
        self.model = SimpleNamespace(modules=lambda: [self.moe, self.moe])

    def _pairs(self):
        return (
            self.experts.mega_l1_weights,
            self.experts.mega_l2_weights,
            self.moe.mega_shared_l1_weights,
            self.moe.mega_shared_l2_weights,
        )

    def test_routed_and_shared_layout_scales_and_parameter_ownership(self):
        before = self._pairs()
        mega_moe.prepare_mega_moe_locality(self.model)
        for original, localized in zip(before, self._pairs()):
            self.assertIs(original[1], localized[1])
            self.assertFalse(localized[0].is_contiguous())
            self.assertNotEqual(original[0].data_ptr(), localized[0].data_ptr())
            restored = localized[0].movedim(0, -3).flatten(-3, -2)
            torch.testing.assert_close(restored, original[0])
        self.assertEqual(
            self.experts.w13_weight.data_ptr(),
            self.experts.mega_l1_weights[0].data_ptr(),
        )
        self.assertEqual(
            self.experts.w2_weight.data_ptr(),
            self.experts.mega_l2_weights[0].data_ptr(),
        )
        self.assertTrue(self.experts._mega_moe_weights_localized)
        self.assertEqual(self.deep_gemm.localize.call_count, 4)
        self.synchronize.assert_called_once_with()
        self.deep_gemm.destroy_localizer.assert_called_once_with()

    def test_idempotent(self):
        mega_moe.prepare_mega_moe_locality(self.model)
        before = self._pairs()
        mega_moe.prepare_mega_moe_locality(self.model)
        for original, current in zip(before, self._pairs()):
            self.assertIs(original, current)
        self.assertEqual(self.deep_gemm.localize.call_count, 4)
        self.deep_gemm.locality_domain.is_localization_available.assert_called_once()

    def test_unfused_shared_experts(self):
        self.moe.mega_shared_l1_weights = None
        self.moe.mega_shared_l2_weights = None
        mega_moe.prepare_mega_moe_locality(self.model)
        self.assertEqual(self.deep_gemm.localize.call_count, 2)
        self.assertTrue(self.experts._mega_moe_weights_localized)
        self.assertIsNone(self.moe.mega_shared_l1_weights)

    def test_unavailable_keeps_original_pairs(self):
        before = self._pairs()
        self.deep_gemm.locality_domain.is_localization_available.return_value = False
        with self.assertLogs(mega_moe.logger, level="WARNING"):
            mega_moe.prepare_mega_moe_locality(self.model)
        for original, current in zip(before, self._pairs()):
            self.assertIs(original, current)
        self.deep_gemm.localize.assert_not_called()
        self.deep_gemm.destroy_localizer.assert_not_called()

    def test_older_deepgemm_without_locality_api(self):
        del self.deep_gemm.locality_domain
        with self.assertLogs(mega_moe.logger, level="WARNING"):
            mega_moe.prepare_mega_moe_locality(self.model)
        self.deep_gemm.localize.assert_not_called()

    def test_unsupported_mma_types_and_eplb(self):
        for flag in ("_mega_moe_w4a4", "_mega_moe_nvfp4"):
            with self.subTest(flag=flag):
                setattr(self.experts, flag, True)
                with self.assertLogs(mega_moe.logger, level="WARNING"):
                    mega_moe.prepare_mega_moe_locality(self.model)
                setattr(self.experts, flag, False)
        self.exec_context.moe.enable_eplb = True
        with self.assertLogs(mega_moe.logger, level="WARNING"):
            mega_moe.prepare_mega_moe_locality(self.model)
        self.deep_gemm.localize.assert_not_called()
        self.deep_gemm.locality_domain.is_localization_available.assert_not_called()

    def test_failed_shared_copy_keeps_layer_consistent_and_releases_localizer(self):
        before = self._pairs()
        before_params = (
            self.experts.w13_weight.data_ptr(),
            self.experts.w2_weight.data_ptr(),
        )
        copy = self.deep_gemm.localize.side_effect
        copies = 0

        def failing_copy(tensor):
            nonlocal copies
            copies += 1
            if copies == 4:
                raise RuntimeError("shared copy failed")
            return copy(tensor)

        self.deep_gemm.localize.side_effect = failing_copy
        with self.assertRaisesRegex(RuntimeError, "shared copy failed"):
            mega_moe.prepare_mega_moe_locality(self.model)
        for original, current in zip(before, self._pairs()):
            self.assertIs(original, current)
        self.assertEqual(
            before_params,
            (
                self.experts.w13_weight.data_ptr(),
                self.experts.w2_weight.data_ptr(),
            ),
        )
        self.assertFalse(getattr(self.experts, "_mega_moe_weights_localized", False))
        self.synchronize.assert_called_once_with()
        self.deep_gemm.destroy_localizer.assert_called_once_with()

    def test_synchronize_failure_still_destroys_localizer(self):
        self.synchronize.side_effect = RuntimeError("sync failed")
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            mega_moe.prepare_mega_moe_locality(self.model)
        self.deep_gemm.destroy_localizer.assert_called_once_with()

    def test_missing_shared_pair_is_rejected_before_allocating(self):
        self.moe.mega_shared_l2_weights = None
        with self.assertRaisesRegex(ValueError, "provided together"):
            mega_moe.prepare_mega_moe_locality(self.model)
        self.deep_gemm.localize.assert_not_called()
        self.deep_gemm.destroy_localizer.assert_called_once_with()

    def test_disabled_or_non_sm100_does_not_probe(self):
        with patch.object(
            mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE_WEIGHTS,
            "get",
            return_value=False,
        ):
            mega_moe.prepare_mega_moe_locality(self.model)
        with patch.object(mega_moe, "_device_sm", 90):
            mega_moe.prepare_mega_moe_locality(self.model)
        self.deep_gemm.locality_domain.is_localization_available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
