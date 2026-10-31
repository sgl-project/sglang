"""Regression test for the Blackwell DeepGEMM UE8M0 weight-scale bug in the
compressed-tensors block-FP8 scheme (sgl-project/sglang#28662).

On Blackwell (SM100) block-wise FP8 dispatches to DeepGEMM, which quantizes
activations to UE8M0 scales and expects the weight scales to be UE8M0-packed
as well. If ``CompressedTensorsW8A8Fp8`` leaves the raw float32 block scales,
DeepGEMM combines UE8M0 activation scales with float32 weight scales and emits
NaN logits. ``process_weights_after_loading`` must requantize the weight scales
to UE8M0 when DeepGEMM is the active runner.

Manual / Blackwell-only: requires SM100 + DeepGEMM (DEEPGEMM_SCALE_UE8M0).
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestCompressedTensorsFp8BlockUE8M0(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_block_fp8_weight_scales_requantized_to_ue8m0(self):
        from sglang.srt.layers import deep_gemm_wrapper

        if not (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        ):
            self.skipTest("requires Blackwell DeepGEMM (DEEPGEMM_SCALE_UE8M0)")

        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationStrategy,
        )

        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
            CompressedTensorsW8A8Fp8,
        )

        torch.manual_seed(0)
        device = "cuda"
        block = [128, 128]
        N, K = 512, 1024  # N % 64 == 0 and K % 128 == 0 -> DeepGEMM-supported shape

        # A bf16 reference weight, block-quantized to fp8 + float32 block scales,
        # exactly as a compressed-tensors `float-quantized` checkpoint stores it.
        w_ref = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1
        w_view = w_ref.float().view(N // 128, 128, K // 128, 128)
        scale = w_view.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0  # (N/128, K/128)
        w_fp8 = (w_view / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(N, K)
        w_deq = (
            (w_fp8.float().view(N // 128, 128, K // 128, 128) * scale[:, None, :, None])
            .view(N, K)
            .to(torch.bfloat16)
        )

        weight_quant = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.BLOCK,
            symmetric=True,
            dynamic=False,
            block_structure=block,
        )
        scheme = CompressedTensorsW8A8Fp8(
            weight_quant=weight_quant, is_static_input_scheme=False
        )

        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        if (
            scheme.w8a8_block_fp8_linear
            is not deepgemm_w8a8_block_fp8_linear_with_fallback
        ):
            self.skipTest("DeepGEMM is not the active block-FP8 runner")

        # Build the layer directly (create_weights would require an initialized
        # tensor-parallel group); the fix lives in process_weights_after_loading.
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            scale.to(torch.float32), requires_grad=False
        )
        layer.orig_dtype = torch.bfloat16

        scheme.process_weights_after_loading(layer)

        # The fix: weight scales must be requantized to UE8M0 for the DeepGEMM runner.
        self.assertTrue(
            getattr(layer.weight_scale, "format_ue8m0", False),
            "block-FP8 weight scales were not requantized to UE8M0",
        )

        x = torch.randn(8, K, device=device, dtype=torch.bfloat16) * 0.1
        y = scheme.apply_weights(layer, x)

        # Pre-fix this path produced NaN; post-fix it matches the bf16 dequant ref.
        self.assertTrue(torch.isfinite(y).all(), "output contains NaN/Inf")
        y_ref = x.float() @ w_deq.float().t()
        rel_err = ((y.float() - y_ref).norm() / y_ref.norm()).item()
        self.assertLess(rel_err, 0.05, f"relative error too high: {rel_err}")


if __name__ == "__main__":
    unittest.main()
