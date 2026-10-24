import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4LinearMethod,
    _linear_nvfp4_scales_to_tma,
)
from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatform

ENV_PATH = (
    "sglang.multimodal_gen.runtime.platforms.cuda.envs."
    "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"
)


def _backend(env_value, *, sm120):
    CudaPlatform.get_modelopt_flashinfer_fp4_backend.cache_clear()
    try:
        with (
            patch(ENV_PATH, env_value),
            patch.object(CudaPlatform, "is_sm120", classmethod(lambda cls: sm120)),
        ):
            return CudaPlatform.get_modelopt_flashinfer_fp4_backend()
    finally:
        CudaPlatform.get_modelopt_flashinfer_fp4_backend.cache_clear()


class TestModeloptFp4BackendDefault(unittest.TestCase):
    """The default FP4 GEMM backend must exist on the device.

    flashinfer.mm_fp4 raises "does not support backend 'trtllm' with capability
    120" on sm_120, so defaulting to trtllm there makes every NVFP4 model fail to
    run at all.
    """

    def test_sm120_does_not_default_to_a_backend_it_lacks(self):
        self.assertEqual(_backend(None, sm120=True), "auto")

    def test_other_gpus_keep_the_trtllm_default(self):
        self.assertEqual(_backend(None, sm120=False), "trtllm")

    def test_explicit_backends_are_passed_through(self):
        for env_value, expected in [
            ("trtllm", "trtllm"),
            ("flashinfer_trtllm", "trtllm"),
            ("cutlass", "cutlass"),
            ("flashinfer_cutlass", "cutlass"),
            ("cudnn", "cudnn"),
            ("flashinfer_cudnn", "cudnn"),
            ("auto", "auto"),
        ]:
            with self.subTest(env=env_value):
                self.assertEqual(_backend(env_value, sm120=True), expected)

    def test_unknown_backend_falls_back_to_the_platform_default(self):
        self.assertEqual(_backend("nonsense", sm120=True), "auto")
        self.assertEqual(_backend("nonsense", sm120=False), "trtllm")


class TestModeloptFp4ScaleProcessing(unittest.TestCase):
    """ModelOptFp4LinearMethod._process_backend_weights / apply scale handling."""

    def test_process_weights_after_loading_matches_tma_layout_independently(self):
        # Independent reference: pad + swizzle to the 128x4 TMA layout by hand,
        # not by calling the helper the production code now calls.
        torch.manual_seed(0)
        # N=32, K=32 (packed K//2=16) satisfies pad_nvfp4_weight's 32-alignment
        # with no padding, keeping this test about the scale layout only.
        weight = torch.randint(0, 255, (32, 16), dtype=torch.uint8, device="cuda")
        scales = torch.randint(0, 255, (16, 8), dtype=torch.uint8, device="cuda")

        from sglang.srt.utils.common import round_up

        M, K = scales.shape
        M_padded, K_padded = round_up(M, 128), round_up(K, 4)
        padded = torch.zeros((M_padded, K_padded), dtype=scales.dtype, device="cuda")
        padded[:M, :K] = scales
        padded = padded.reshape(M_padded // 128, 4, 32, K_padded // 4, 4)
        padded = padded.permute(0, 3, 2, 1, 4).contiguous()
        expected = padded.reshape(M_padded, K_padded)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
        layer.weights_padding_cols = 0
        with patch(
            "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant."
            "_get_fp4_gemm_op",
            return_value=(None, "cutlass"),
        ):
            method = ModelOptFp4LinearMethod.__new__(ModelOptFp4LinearMethod)
            method.quant_config = MagicMock(swap_weight_nibbles=False)
            method._process_backend_weights(layer, scales)

        self.assertTrue(torch.equal(layer.weight_scale_interleaved, expected))
        self.assertTrue(layer.weight_scale_interleaved.is_cuda)


if __name__ == "__main__":
    unittest.main()
