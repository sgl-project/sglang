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

import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
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

        with patch.object(
            fp8_utils,
            "_get_flashinfer_groupwise_backend",
            return_value="trtllm",
            create=True,
        ), patch.object(
            fp8_utils, "gemm_fp8_nt_groupwise", trtllm_spy, create=True
        ), patch.object(
            fp8_utils, "triton_w8a8_block_fp8_linear", triton_spy
        ), patch.object(
            fp8_utils, "sglang_per_token_group_quant_fp8", quant_spy
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
