"""CPU checks for avoiding the unused Torch mHC-pre wrapper."""

import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _MhcOps:
    def __init__(self):
        self.mix = None

    def mhc_pre(self, *, residual, **kwargs):
        rows = residual.shape[0]
        device = residual.device
        post = torch.ones((rows, 2, 1), dtype=torch.float32, device=device)
        comb = torch.zeros((rows, 2, 2), dtype=torch.float32, device=device)
        return post, comb, residual.mean(dim=1) + 1

    def hc_split_sinkhorn(self, mixes, *args):
        self.mix = mixes
        rows = mixes.shape[0]
        device = mixes.device
        pre = torch.zeros((rows, 1, 2), dtype=torch.float32, device=device)
        post = torch.ones_like(pre)
        comb = torch.zeros((rows, 1, 2, 2), dtype=torch.float32, device=device)
        return pre, post, comb


def _compile_passthrough(fn):
    def compiled(*args, **kwargs):
        return fn(*args, **kwargs)

    return compiled


def _combine_cpu(x_flat, _pre, hc_mult, dtype):
    return x_flat.reshape(x_flat.shape[0], hc_mult, -1).mean(dim=1).to(dtype)


class TestDeepseekV4HcPreWrapper(CustomTestCase):
    def setUp(self):
        self.x = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
            ]
        )
        self.hc_fn = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        self.hc_scale = torch.eye(2)
        self.hc_base = torch.zeros((2, 2))
        self.model = types.SimpleNamespace(
            rms_norm_eps=1e-5,
            hc_mult=2,
            hc_eps=1e-6,
            hc_sinkhorn_iters=2,
        )
        self.ops = _MhcOps()
        self.runtime_patches = (
            patch.object(deepseek_v4, "_is_npu", False),
            patch.object(deepseek_v4, "_is_xpu", False),
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "_is_cuda", True),
            patch.object(deepseek_v4, "is_allocation_symmetric", return_value=False),
            patch.object(
                deepseek_v4, "use_symmetric_memory", return_value=nullcontext()
            ),
            patch.object(deepseek_v4, "get_mhc_ops", return_value=self.ops),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=types.SimpleNamespace(tp_group=object()),
            ),
        )

    def _enter_runtime_patches(self):
        for patcher in self.runtime_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _call(self, norm=None):
        return deepseek_v4.DeepseekV4DecoderLayer.hc_pre(
            self.model,
            self.x,
            self.hc_fn,
            self.hc_scale,
            self.hc_base,
            norm=norm,
        )

    def test_tilelang_branch_skips_fallback_definition_and_preserves_result(self):
        self._enter_runtime_patches()
        tilelang_ops = types.ModuleType("sglang.kernels.ops.layernorm.mhc")
        tilelang_ops.mhc_pre = self.ops.mhc_pre
        with (
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.layernorm.mhc": tilelang_ops},
            ),
            envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch("torch.compile", side_effect=_compile_passthrough) as compile_fn,
            patch.object(deepseek_v4, "compile_in_capture_mode") as compile_decorator,
        ):
            result = self._call()

        self.assertEqual(len(result), 4)
        torch.testing.assert_close(result[0], self.x.mean(dim=1) + 1)
        self.assertEqual(result[1].shape, (2, 2))
        self.assertEqual(result[2].shape, (2, 2, 2))
        self.assertFalse(result[3])
        compile_decorator.assert_not_called()
        compile_fn.assert_not_called()
        self.assertIsNone(self.ops.mix)

    def test_empty_and_xpu_paths_skip_torch_fallback_definition(self):
        self._enter_runtime_patches()
        with (
            envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            patch("torch.compile", side_effect=_compile_passthrough) as compile_fn,
            patch.object(deepseek_v4, "compile_in_capture_mode") as compile_decorator,
        ):
            self.x = self.x[:0]
            empty_result = self._call()
            self.assertEqual(empty_result[0].shape, (0, 3))

            self.x = torch.ones((2, 2, 3))
            with patch.object(deepseek_v4, "_is_xpu", True):
                xpu_result = self._call()

        self.assertEqual(xpu_result[0].shape, (2, 3))
        compile_decorator.assert_not_called()
        compile_fn.assert_not_called()

    def test_fallback_keeps_torch_result_and_eager_decorator_is_identity(self):
        self._enter_runtime_patches()
        with (
            envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False),
            patch.object(capture_mode, "is_capture_mode", False),
            patch.object(capture_mode, "is_gfx1250_supported", return_value=False),
            patch("torch.compile", side_effect=_compile_passthrough) as compile_fn,
            patch(
                "sglang.kernels.ops.layernorm.mhc.mhc_combine", side_effect=_combine_cpu
            ),
        ):
            result = self._call()

        x_flat = self.x.flatten(1).float()
        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.model.rms_norm_eps
        )
        expected_mix = (
            torch.nn.functional.linear(x_flat, self.hc_fn) * rsqrt
        ).unsqueeze(1)
        torch.testing.assert_close(self.ops.mix, expected_mix)
        torch.testing.assert_close(result[0], self.x.mean(dim=1).float())
        torch.testing.assert_close(result[1], torch.ones((2, 2)))
        self.assertEqual(result[2].shape, (2, 2, 2))
        self.assertFalse(result[3])
        compile_fn.assert_not_called()

    def test_capture_mode_fallback_still_uses_compiled_helper(self):
        self._enter_runtime_patches()
        with (
            envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False),
            patch.object(capture_mode, "is_capture_mode", True),
            patch.object(capture_mode, "is_gfx1250_supported", return_value=False),
            patch("torch.compile", side_effect=_compile_passthrough) as compile_fn,
            patch(
                "sglang.kernels.ops.layernorm.mhc.mhc_combine", side_effect=_combine_cpu
            ),
        ):
            result = self._call()

        compile_fn.assert_called_once()
        self.assertIsNotNone(self.ops.mix)
        self.assertTrue(torch.isfinite(result[0]).all())
        torch.testing.assert_close(result[0], self.x.mean(dim=1).float())


if __name__ == "__main__":
    unittest.main()
