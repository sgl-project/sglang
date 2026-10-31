"""On ROCm without aiter or vLLM, GemmaRMSNorm must use the Triton Gemma kernels
when they import; it used to fall back to forward_native, and no setting
without aiter could select the kernels."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import layernorm
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 8


class TestGemmaRMSNormHipDispatch(CustomTestCase):
    def _forward_without_aiter_or_vllm(self, *, has_triton_kernels):
        triton_norm = MagicMock(return_value="triton")
        with (
            patch.object(layernorm, "_use_aiter", False),
            patch.object(layernorm, "_has_vllm_rms_norm", False),
            patch.object(
                layernorm, "_has_rocm_triton_gemma_rms_norm", has_triton_kernels
            ),
            patch.object(
                layernorm, "rocm_triton_gemma_rmsnorm", triton_norm, create=True
            ),
            patch.object(
                GemmaRMSNorm, "forward_native", MagicMock(return_value="native")
            ),
        ):
            return GemmaRMSNorm(HIDDEN).forward_hip(torch.randn(2, HIDDEN))

    def test_triton_kernels_are_used_without_aiter_or_vllm(self):
        self.assertEqual(
            self._forward_without_aiter_or_vllm(has_triton_kernels=True), "triton"
        )

    def test_native_fallback_when_the_triton_kernels_do_not_import(self):
        self.assertEqual(
            self._forward_without_aiter_or_vllm(has_triton_kernels=False), "native"
        )


if __name__ == "__main__":
    unittest.main()
