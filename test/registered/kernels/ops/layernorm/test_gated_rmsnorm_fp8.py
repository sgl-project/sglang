import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
from sglang.kernels.ops.layernorm import gated_rmsnorm_static_fp8
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear_bmm_flashinfer,
    flashinfer_per_tensor_fp8_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGatedRmsnormStaticFp8(CustomTestCase):
    def _case(
        self,
        tokens,
        heads,
        scale=0.02,
        flat=False,
        weight_dtype=torch.bfloat16,
        extreme=False,
    ):
        torch.manual_seed(119)
        x = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        packed = torch.randn(tokens, heads * 320 + 32, device="cuda", dtype=x.dtype)
        z = packed[:, : heads * 128].view(tokens, heads, 128)
        weight = torch.randn(128, device="cuda", dtype=weight_dtype)
        if extreme:
            # Zero rows, large gates, saturation, and adjacent BF16 values.
            x[0] = 0
            x[1] = torch.linspace(-2, 2, 128, device="cuda").to(x.dtype)
            z[1] = 20
            weight *= 16
        static_scale = torch.tensor([scale], device="cuda", dtype=torch.float32)
        scale_before = static_scale.clone()
        supplied = torch.empty(
            tokens, heads * 128, device="cuda", dtype=torch.float8_e4m3fn
        )
        actual = gated_rmsnorm_static_fp8(
            x.view(-1, 128) if flat else x,
            z,
            weight,
            static_scale,
            out=supplied,
        )
        self.assertIs(actual, supplied)
        if tokens:
            expected = torch.empty_like(actual)
            # Bound reference indexing while the candidate sees the full packed
            # projection, including offsets beyond signed 32-bit element counts.
            for start in range(0, tokens, 32768):
                stop = min(start + 32768, tokens)
                norm, _, _ = _layer_norm_fwd(
                    x[start:stop].reshape(-1, 128),
                    weight,
                    None,
                    1e-6,
                    z=z[start:stop],
                    is_rms_norm=True,
                )
                quant, _ = static_quant_fp8(norm.view(stop - start, -1), static_scale)
                expected[start:stop].copy_(quant)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )
        self.assertTrue(torch.equal(static_scale, scale_before))

    def test_rounding_and_strided_gate(self):
        """Preserve materialized BF16 rounding, static scale and packed gate strides."""
        for tokens in (0, 1, 2, 3, 63, 64, 65):
            with self.subTest(tokens=tokens):
                self._case(
                    tokens,
                    8,
                    flat=tokens % 2 == 0,
                    weight_dtype=torch.float32 if tokens % 2 else torch.bfloat16,
                )
        self._case(65, 16, scale=0.0001, flat=True, extreme=True)
        self._case(65, 8, scale=0.037, weight_dtype=torch.float32, extreme=True)

    def test_single_row_reduction_rounding(self):
        """A rounding-boundary row distinguishes the reference reduction layouts."""
        x_bits = (
            "f9bfa73f6dbf2d3f1c3f783f2c3ea73e55bf973d33407b3e653f044019bf17bf"
            "d0be95be92bfd03db0bf8f3fc0bd03bf363f39bf33bd303f283f90bf40bf15bf"
            "85bf313f823e90bfcb3f1a3f93bf903ffcbf23bf11bf2fbf3bbe2f3f52be11bf"
            "a7be3c3f9abf7f3e1cbf5ebf09407abfd8bef8be633f34bed73fa23fe2bd113d"
            "98beed3e343ee3bf873cc4bf4140e33eb73e4bbf04c0453d1f3f9dbf803e823e"
            "4dbf06c087bc90bf653f62bf9abef13e08bf053f5b3f903e953e913e8b3f1c3f"
            "043f923f003fcc3d01bff3bf1bbfd8bee53ea5bf45bf513efebeb53e7e3f963f"
            "f53ea43f2c3e183fc0be1a3f92be21bf123f6e3f88bfb7bd94bf0cbf02c084be"
        )
        z_bits = (
            "b93f7e3eeb3fcc3de8bf763e47bfb73ecf3f04bcef3e36bfbebeecbdfa3f873f"
            "b13d55bfe53d2f3f0840ebbdb73e893f213fe5be793fbdbf053fd83f12bf58bf"
            "11bfff3e383fb5bd7fbf70bf7a3f2ac05c3f8c3ff5bef9bfc33f0bc0043f87bf"
            "493fe0bff0be83bfcf3d17bf89bfeebf16be81bff63a283e22bf5d3f743f843e"
            "babd6f3e473ee03f0c3fd83faa3f0940403fc43fd7bfe1be0bbea23f4e3d8ebe"
            "60bf9ebf20c0cebdce3eb73f1bbf1a3ea6bf843e10bf33bf36bf73bf22bf7bbe"
            "b7bfc23fd4bf74befd3f28bf95bfde3f88be163f393f253f343f9e3f08bf5ebe"
            "4b3fa7be53bff3bdacbfa2bead3fd4be95be0040313e423e58bff6bf93bf40bf"
        )
        weight_bits = (
            "a0bf10c05d3fe33e4f3f3d3f51beeabce13fd03e0e3f22bc943f80be243f96be"
            "963f853fc7bfd8beaf3facbe6b3e023f143f32bf07404cbf943f9abeb23f763f"
            "27bfa63e9fbef0befabf0dbf10be583fb8bf61bf913e1a3ed4beffbf94bf23bf"
            "283fac3dd13f4ebfff3e20beb33f313fb4bf8f3edebd8ebf9b3f8d3f7fbd58bd"
            "053f343f05c0a0bffebdc73fb0be8bbe0740923fb4bfda3f1d3fa83c20be35bf"
            "943eaf3f623f233f084061bf3c3fe23f00405d3fa7bfc73e0fbe12bf783f1040"
            "483f393f103d643ff4be6fbf1440a3bf3cbe62c0913e9abe363f623f37be94bf"
            "fabf263f413f8d3d8b3e643f493fbbbfa03e4fbe91be38bfb4bff23f1e3c363e"
        )

        def restore(bits):
            return torch.frombuffer(
                bytearray.fromhex(bits), dtype=torch.bfloat16
            ).cuda()

        x = restore(x_bits).repeat(8).view(1, 8, 128)
        z = restore(z_bits).repeat(8).view(1, 8, 128)
        weight = restore(weight_bits)
        scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        for invariant, piecewise in ((False, False), (True, False), (False, True)):
            with (
                self.subTest(invariant=invariant, piecewise=piecewise),
                patch(
                    "sglang.kernels.ops.attention.fla.layernorm_gated.is_batch_invariant_mode_enabled",
                    return_value=invariant,
                ),
                patch(
                    "sglang.kernels.ops.attention.fla.layernorm_gated.check_cuda_graph_backend",
                    return_value=piecewise,
                ),
            ):
                expected, _, _ = _layer_norm_fwd(
                    x.view(-1, 128), weight, None, 1e-6, z=z, is_rms_norm=True
                )
                expected, _ = static_quant_fp8(expected.view(1, -1), scale)
                actual = gated_rmsnorm_static_fp8(x, z, weight, scale)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )

    def test_long_prefill_head_splits(self):
        """Long batches preserve the same per-head reduction for all head splits."""
        for tokens, heads in ((4096, 64), (32768, 16), (131072, 8), (131072, 64)):
            with self.subTest(tokens=tokens, heads=heads):
                self._case(tokens, heads)
        self._case(4096, 64, weight_dtype=torch.float32)

    @unittest.skipUnless(
        flashinfer_per_tensor_fp8_supported(), "FlashInfer per-tensor FP8 bmm"
    )
    def test_prequantized_input_matches_flashinfer_bmm(self):
        """The fused norm's FP8 output must give the out_proj GEMM the bits of its own static quant."""
        torch.manual_seed(7)
        tokens, heads, hidden = 333, 16, 1024
        x = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        z = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        input_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        w = torch.randn(hidden, heads * 128, device="cuda", dtype=torch.bfloat16)
        w_fp8 = (w * 32).to(torch.float8_e4m3fn).t()
        w_scale = torch.tensor([0.03], device="cuda", dtype=torch.float32)
        norm, _, _ = _layer_norm_fwd(
            x.view(-1, 128), weight, None, 1e-6, z=z, is_rms_norm=True
        )
        expected = apply_fp8_linear_bmm_flashinfer(
            norm.view(tokens, -1), w_fp8, w_scale, input_scale
        )
        fp8 = gated_rmsnorm_static_fp8(x, z, weight, input_scale)
        actual = apply_fp8_linear_bmm_flashinfer(fp8, w_fp8, w_scale, input_scale)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
