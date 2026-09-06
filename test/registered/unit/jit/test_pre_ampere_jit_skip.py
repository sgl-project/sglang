"""Pre-Ampere CUDA JIT short-circuit and FP8 capability alignment."""

from unittest.mock import patch

from sglang.kernels.jit.utils.arch import is_pre_ampere_cuda
from sglang.kernels.ops.diffusion.rope.qknorm_rope_jit import (
    can_use_fused_inplace_qknorm_rope,
)
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
import torch

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPreAmpereJitSkip(CustomTestCase):
    @patch("sglang.kernels.jit.utils.arch.torch.cuda.is_available", return_value=True)
    @patch(
        "sglang.kernels.jit.utils.arch.torch.cuda.get_device_capability",
        return_value=(7, 5),
    )
    def test_is_pre_ampere_cuda_turing(self, _cap, _avail):
        self.assertTrue(is_pre_ampere_cuda())

    @patch("sglang.kernels.jit.utils.arch.torch.cuda.is_available", return_value=True)
    @patch(
        "sglang.kernels.jit.utils.arch.torch.cuda.get_device_capability",
        return_value=(8, 0),
    )
    def test_is_pre_ampere_cuda_ampere(self, _cap, _avail):
        self.assertFalse(is_pre_ampere_cuda())

    @patch(
        "sglang.kernels.ops.diffusion.rope.qknorm_rope_jit.is_pre_ampere_cuda",
        return_value=True,
    )
    def test_qknorm_rope_skips_jit_probe_on_pre_ampere(self, _pre):
        self.assertFalse(
            can_use_fused_inplace_qknorm_rope(
                128,
                128,
                True,
                torch.bfloat16,
                torch.float32,
            )
        )

    @patch(
        "sglang.srt.layers.quantization.fp8_utils.cutlass_fp8_supported",
        return_value=True,
    )
    def test_cuda_supports_fp8_follows_cutlass(self, _cutlass):
        self.assertTrue(CudaSRTPlatform().supports_fp8())

    @patch(
        "sglang.srt.layers.quantization.fp8_utils.cutlass_fp8_supported",
        return_value=False,
    )
    def test_cuda_supports_fp8_false_on_turing(self, _cutlass):
        self.assertFalse(CudaSRTPlatform().supports_fp8())
