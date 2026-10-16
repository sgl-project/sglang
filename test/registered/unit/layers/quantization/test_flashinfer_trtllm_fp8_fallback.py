"""Unit test for the FlashInfer TRTLLM block-FP8 GEMM fallback decision.

Regression guard for two coupled behaviors in
``flashinfer_gemm_w8a8_block_fp8_linear_with_fallback``:

* DeepSeek-R1 perf regression (commit 5da265de): with
  ``--fp8-gemm-backend flashinfer_trtllm`` the dense block-FP8 weight scales are
  plain float32 (they are NOT requantized to UE8M0 -- that only happens on the
  DeepGEMM dispatch path). The TRTLLM groupwise GEMM consumes float32 scales, so
  a bf16 layer must use the TRTLLM kernel, not fall back to triton. Gating the
  fallback on a ``format_ue8m0`` weight-scale attribute wrongly forced every such
  layer onto the slow triton path.
* MiniMax-M2.5 accuracy fix (PR #22300): the TRTLLM GEMM is only numerically
  correct for bf16 output, so fp16 output must fall back to triton.

So the fallback must key on output dtype and K (>= 256), independent of any
``format_ue8m0`` scale attribute. These tests pin that exactly, mocking the
backend selector and the two GEMM implementations so they run on CPU CI.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import functools
import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.quantization.fp8 as fp8
import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    dispatch_w8a8_block_fp8_linear,
    triton_w8a8_block_fp8_linear,
)
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = [128, 128]
M = 16
N = 512


class TestFlashinferTrtllmFp8Fallback(CustomTestCase):
    def _invoke(self, dtype, k, *, set_format_ue8m0=False):
        """Call the fallback dispatcher with backend pinned to 'trtllm'.

        Returns (triton_spy, trtllm_spy) so callers can assert which path ran.
        Every GEMM implementation is mocked, so no kernels actually execute.
        """
        input_2d = torch.zeros((M, k), dtype=dtype)
        weight = torch.zeros((N, k), dtype=torch.float32)
        weight_scale = torch.zeros((N // 128, k // 128), dtype=torch.float32)
        if set_format_ue8m0:
            # Pre-fix, this attribute is what gated the trtllm path. It must now
            # be irrelevant: a bf16 layer uses trtllm whether or not it is set.
            weight_scale.format_ue8m0 = True

        triton_spy = MagicMock(return_value=torch.zeros((M, N), dtype=dtype))
        trtllm_spy = MagicMock(return_value=torch.zeros((M, N), dtype=dtype))
        quant_spy = MagicMock(return_value=(MagicMock(), MagicMock()))

        with (
            patch.object(
                fp8_utils,
                "_get_flashinfer_groupwise_backend",
                return_value="trtllm",
                create=True,
            ),
            patch.object(fp8_utils, "gemm_fp8_nt_groupwise", trtllm_spy, create=True),
            patch.object(fp8_utils, "triton_w8a8_block_fp8_linear", triton_spy),
            patch.object(fp8_utils, "sglang_per_token_group_quant_fp8", quant_spy),
        ):
            fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
                input_2d, weight, BLOCK_SIZE, weight_scale
            )
        return triton_spy, trtllm_spy

    def test_bf16_uses_trtllm_with_plain_fp32_scales(self):
        """DeepSeek-R1 regression guard: bf16 + K>=256 + plain fp32 scales
        (no format_ue8m0) must use the trtllm GEMM, not fall back to triton."""
        triton_spy, trtllm_spy = self._invoke(torch.bfloat16, 512)
        trtllm_spy.assert_called_once()
        triton_spy.assert_not_called()

    def test_bf16_uses_trtllm_regardless_of_format_ue8m0(self):
        """format_ue8m0 must not affect the decision: bf16 still uses trtllm."""
        triton_spy, trtllm_spy = self._invoke(
            torch.bfloat16, 512, set_format_ue8m0=True
        )
        trtllm_spy.assert_called_once()
        triton_spy.assert_not_called()

    def test_fp16_falls_back_to_triton(self):
        """MiniMax-M2.5 accuracy guard: fp16 output must fall back to triton."""
        triton_spy, trtllm_spy = self._invoke(torch.float16, 512)
        triton_spy.assert_called_once()
        trtllm_spy.assert_not_called()

    def test_small_k_falls_back_to_triton(self):
        """K < 256 is unsupported by the trtllm GEMM and must fall back."""
        triton_spy, trtllm_spy = self._invoke(torch.bfloat16, 128)
        triton_spy.assert_called_once()
        trtllm_spy.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=3)


class TestBlockSizeDispatch(CustomTestCase):
    """Non-128-wide K blocks dispatch to Triton regardless of --fp8-gemm-backend;
    128-wide blocks keep the backend choice."""

    def test_128_wide_k_blocks_keep_the_backend_choice(self):
        for name in ("triton", "deep_gemm"):
            with (
                self.subTest(backend=name),
                patch.object(
                    fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(name)
                ),
            ):
                default = dispatch_w8a8_block_fp8_linear()
                self.assertIs(dispatch_w8a8_block_fp8_linear([128, 128]), default)
                self.assertIs(dispatch_w8a8_block_fp8_linear([1, 128]), default)
                fn = dispatch_w8a8_block_fp8_linear([32, 32], act_scale_ue8m0=True)
                self.assertIsInstance(fn, functools.partial)
                self.assertIs(fn.func, triton_w8a8_block_fp8_linear)
                self.assertEqual(fn.keywords, {"act_scale_ue8m0": True})

    def test_method_dispatches_on_the_effective_block_size(self):
        # An MXFP8 checkpoint converted to block-fp8 at load time is a [128, 128]
        # weight; dispatching on the pre-conversion [1, 32] would pick Triton.
        with (
            patch.object(
                fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend.TRITON
            ),
            patch.object(fp8, "_mxfp8_to_block_fp8_required", True),
        ):
            method = Fp8LinearMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    use_mxfp8=True,
                    weight_block_size=[1, 32],
                    scale_fmt="ue8m0",
                )
            )
            self.assertTrue(method.convert_mxfp8_to_block)
            self.assertIs(method.w8a8_block_fp8_linear, triton_w8a8_block_fp8_linear)
            self.assertFalse(method.block_fp8_as_mxfp8)


class TestFlashinferPrequantizedInput(CustomTestCase):
    """A pre-quantized ``(fp8 q, (m, k // 128) scales)`` input goes to the GEMM
    as is, in the scale layout each FlashInfer backend reads, and never through
    the quantizer again."""

    def _invoke(self, backend, k, input_scale):
        q = torch.zeros((M, k), dtype=torch.float8_e4m3fn)
        weight = torch.zeros((N, k), dtype=torch.float32)
        weight_scale = torch.zeros((N // 128, k // 128), dtype=torch.float32)

        triton_spy = MagicMock(return_value=torch.zeros((M, N), dtype=torch.bfloat16))
        gemm_spy = MagicMock(return_value=torch.zeros((M, N), dtype=torch.bfloat16))
        quant_spy = MagicMock()

        with (
            patch.object(
                fp8_utils,
                "_get_flashinfer_groupwise_backend",
                return_value=backend,
                create=True,
            ),
            patch.object(fp8_utils, "gemm_fp8_nt_groupwise", gemm_spy, create=True),
            patch.object(fp8_utils, "triton_w8a8_block_fp8_linear", triton_spy),
            patch.object(fp8_utils, "sglang_per_token_group_quant_fp8", quant_spy),
        ):
            out = fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
                q, weight, BLOCK_SIZE, weight_scale, input_scale=input_scale
            )
        quant_spy.assert_not_called()
        self.assertEqual(out.dtype, torch.bfloat16)
        return q, triton_spy, gemm_spy

    @staticmethod
    def _scales(k, *, column_major):
        scales = torch.arange(M * (k // 128), dtype=torch.float32).view(M, k // 128)
        return scales.t().contiguous().t() if column_major else scales

    def test_cutlass_takes_mn_major_scales(self):
        for column_major in (False, True):
            with self.subTest(column_major=column_major):
                scales = self._scales(512, column_major=column_major)
                q, triton_spy, gemm_spy = self._invoke("cutlass", 512, scales)
                triton_spy.assert_not_called()
                (a, _, x_scale, _), kwargs = gemm_spy.call_args
                self.assertEqual(a.data_ptr(), q.data_ptr())
                self.assertTrue(x_scale.is_contiguous())
                self.assertTrue(torch.equal(x_scale, scales.t()))
                self.assertEqual(kwargs["out_dtype"], torch.bfloat16)

    def test_trtllm_takes_column_major_scales(self):
        for column_major in (False, True):
            with self.subTest(column_major=column_major):
                scales = self._scales(512, column_major=column_major)
                q, triton_spy, gemm_spy = self._invoke("trtllm", 512, scales)
                triton_spy.assert_not_called()
                (a, _, x_scale, _), kwargs = gemm_spy.call_args
                self.assertEqual(a.data_ptr(), q.data_ptr())
                self.assertEqual(x_scale.stride(), (1, M))
                self.assertTrue(torch.equal(x_scale, scales))
                self.assertEqual(kwargs["out_dtype"], torch.bfloat16)

    def test_trtllm_small_k_falls_back_with_row_major_scales(self):
        scales = self._scales(128, column_major=True)
        q, triton_spy, gemm_spy = self._invoke("trtllm", 128, scales)
        gemm_spy.assert_not_called()
        (a, _, _, _, x_scale, _), _ = triton_spy.call_args
        self.assertEqual(a.data_ptr(), q.data_ptr())
        self.assertTrue(x_scale.is_contiguous())
        self.assertTrue(torch.equal(x_scale, scales))

    def test_rejects_scales_of_the_wrong_shape(self):
        with self.assertRaises(AssertionError):
            self._invoke("cutlass", 512, torch.zeros((M, 3), dtype=torch.float32))
