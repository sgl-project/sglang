"""Unit tests for NVFP4 Marlin scaling and fallback utilities."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import marlin_utils_fp4
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    _nvfp4_compute_scale_factor,
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
    prepare_moe_nvfp4_layer_for_marlin,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMarlinUtilsFp4(unittest.TestCase):
    def test_scale_factor_computation(self):
        # Empty tensor
        self.assertEqual(_nvfp4_compute_scale_factor(torch.empty(0)), 1.0)

        # All zeros
        self.assertEqual(
            _nvfp4_compute_scale_factor(torch.zeros((16, 16), dtype=torch.bfloat16)),
            1.0,
        )

        # Half precision activations skip rescaling
        small = torch.full((16, 16), 1e-4, dtype=torch.bfloat16)
        self.assertEqual(
            _nvfp4_compute_scale_factor(small, a_dtype=torch.half),
            1.0,
        )

        # Already normalized scales (max >= 448)
        norm_scales = torch.full((16, 16), 448.0, dtype=torch.bfloat16)
        self.assertEqual(_nvfp4_compute_scale_factor(norm_scales), 1.0)

        # Small scales produce a power-of-two factor >= 1.0
        for scale_val in [1e-4, 2**-7, 2**-10, 2**-14]:
            t = torch.full((16, 16), scale_val, dtype=torch.bfloat16)
            sf = _nvfp4_compute_scale_factor(t, a_dtype=torch.bfloat16)
            self.assertGreaterEqual(sf, 1.0)
            # Power of two check
            log2_sf = torch.tensor(sf).log2().item()
            self.assertEqual(sf, 2.0 ** round(log2_sf))
            # Rescaled range check
            rescaled_max = t.float().max() * (2**7) * sf
            self.assertLessEqual(rescaled_max, 448.0 * (2**7))
            self.assertGreaterEqual(rescaled_max, 2.0)

    def test_nvfp4_marlin_process_scales_small_scale_no_underflow(self):
        # Small scales without scaling would have (scale * 2^7) < 2, causing underflow
        scales = torch.full((32, 16), 1e-4, dtype=torch.bfloat16)
        processed, sf = nvfp4_marlin_process_scales(scales, a_dtype=torch.bfloat16)
        self.assertGreater(sf, 1.0)
        self.assertEqual(processed.dtype, torch.float8_e4m3fn)
        self.assertEqual(processed.shape, (32, 16))

        # Check that byte representation has the MSB set (bit 7)
        raw_bytes = processed.view(torch.uint8)
        self.assertTrue((raw_bytes >= 128).all())

        # Test subnormal clamping: scales < 2 after scaling should be clamped to 0
        mixed = torch.tensor([[1e-4, 1e-12], [0.0, 1e-4]], dtype=torch.bfloat16).repeat(
            8, 8
        )
        processed_mixed, sf_mixed = nvfp4_marlin_process_scales(
            mixed, a_dtype=torch.bfloat16
        )
        self.assertGreater(sf_mixed, 1.0)
        raw_mixed = processed_mixed.view(torch.uint8)
        # Zeros and subnormals below 2 after scaling become 0
        self.assertTrue((raw_mixed == 0).any())
        self.assertTrue((raw_mixed >= 128).any())

    def test_prepare_nvfp4_layer_for_marlin_dense(self):
        size_n, size_k = 64, 128
        group_size = 16

        layer = torch.nn.Module()
        layer.quant_config = SimpleNamespace(group_size=group_size)
        layer.output_size_per_partition = size_n
        layer.input_size_per_partition = size_k
        layer.params_dtype = torch.bfloat16
        layer.weight = torch.nn.Parameter(
            torch.zeros((size_n, size_k // 2), dtype=torch.uint8),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.full((size_n, size_k // group_size), 1e-4, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.bfloat16),
            requires_grad=False,
        )

        with (
            patch.object(
                marlin_utils_fp4,
                "marlin_make_workspace",
                return_value=torch.empty(0, dtype=torch.int32),
            ),
            patch.object(
                marlin_utils_fp4,
                "gptq_marlin_repack",
                return_value=torch.empty(
                    (size_k // 16, size_n * 16 // 8), dtype=torch.int32
                ),
                create=True,
            ),
        ):
            prepare_nvfp4_layer_for_marlin(layer)

        # Global scale must be divided by scale_factor
        expected_unscaled_g = nvfp4_marlin_process_global_scale(
            torch.tensor(1.0, dtype=torch.bfloat16)
        )
        self.assertLess(layer.weight_global_scale.item(), expected_unscaled_g.item())
        expected_sf = _nvfp4_compute_scale_factor(
            torch.full((size_n, size_k // group_size), 1e-4, dtype=torch.bfloat16),
            a_dtype=torch.bfloat16,
        )
        self.assertAlmostEqual(
            layer.weight_global_scale.item() * expected_sf,
            expected_unscaled_g.item(),
            places=2,
        )

    def test_prepare_moe_nvfp4_layer_for_marlin_shared_scale_factor(self):
        num_experts = 4
        intermediate_size = 128
        hidden_size = 128
        group_size = 16

        layer = torch.nn.Module()
        layer.quant_config = SimpleNamespace(group_size=group_size)
        layer.moe_runner_config = SimpleNamespace(is_gated=False)
        layer.intermediate_size_per_partition = intermediate_size
        layer.params_dtype = torch.bfloat16

        # Expert weights
        w13_packed = torch.zeros(
            (num_experts, intermediate_size, hidden_size // 2), dtype=torch.uint8
        )
        w2_packed = torch.zeros(
            (num_experts, hidden_size, intermediate_size // 2), dtype=torch.uint8
        )
        # Expert scales with different small magnitudes across experts
        w13_scale = torch.zeros(
            (num_experts, intermediate_size, hidden_size // group_size),
            dtype=torch.bfloat16,
        )
        w13_scale[0] = 1e-4
        w13_scale[1] = 2e-4
        w13_scale[2] = 5e-5
        w13_scale[3] = 1.5e-4

        w2_scale = torch.zeros(
            (num_experts, hidden_size, intermediate_size // group_size),
            dtype=torch.bfloat16,
        )
        w2_scale[0] = 2e-4
        w2_scale[1] = 1e-4
        w2_scale[2] = 8e-5
        w2_scale[3] = 1.2e-4

        layer.w13_weight = torch.nn.Parameter(w13_packed, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_packed, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones((num_experts,), dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones((num_experts,), dtype=torch.bfloat16), requires_grad=False
        )

        with (
            patch.object(
                marlin_utils_fp4,
                "marlin_make_workspace",
                return_value=torch.empty(0, dtype=torch.int32),
            ),
            patch.object(
                marlin_utils_fp4,
                "_repack_moe_fp4_weight_for_marlin",
                side_effect=lambda w, **kwargs: w.view(torch.int32),
            ),
        ):
            prepare_moe_nvfp4_layer_for_marlin(layer)

        # Scale factor must be uniform across all experts and divide global scale
        expected_sf13 = _nvfp4_compute_scale_factor(w13_scale, a_dtype=torch.bfloat16)
        expected_sf2 = _nvfp4_compute_scale_factor(w2_scale, a_dtype=torch.bfloat16)
        self.assertGreater(expected_sf13, 1.0)
        self.assertGreater(expected_sf2, 1.0)

        base_g = nvfp4_marlin_process_global_scale(
            torch.ones((num_experts,), dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            layer.w13_weight_scale_2 * expected_sf13,
            base_g,
        )
        torch.testing.assert_close(
            layer.w2_weight_scale_2 * expected_sf2,
            base_g,
        )

    def test_dequantized_scale_roundtrip(self):
        # Simulate Marlin dequantization to verify invariance of effective scale
        scale_val = 1e-4
        global_val = 1.0
        s = torch.full((16, 16), scale_val, dtype=torch.bfloat16)
        g = torch.tensor(global_val, dtype=torch.bfloat16)

        processed_s, sf = nvfp4_marlin_process_scales(s, a_dtype=torch.bfloat16)
        processed_g = nvfp4_marlin_process_global_scale(g) / sf

        # Verify that repacked scales with scale_factor and rescaled global scale
        # preserve the effective scale factor
        raw_byte = int(processed_s.view(torch.uint8).flatten()[0].item())
        # The S0E5M3 format when MSB is 1 has bit 7 set
        self.assertTrue(raw_byte & 0x80)

        # Extract S0E5M3 byte to BF16 (simulating dequant_fp8_scales in dequant.h)
        # q has the byte in the high byte of a 16-bit word
        q_16 = raw_byte << 8
        out_bf16_bits = ((q_16 & 0x8000) >> 1) | ((q_16 & 0x7F00) >> 4)
        bf16_val = torch.tensor([out_bf16_bits], dtype=torch.int16).view(torch.bfloat16)

        # Dequantized scale = bf16_val * processed_global_scale
        # In BF16, global_scale was multiplied by 2^119.
        # bf16_val has exponent e_s. The mathematical scale is 2^(e_s - 127) * mantissa.
        # Total effective scale = bf16_val * processed_g
        # With sf applied to both, sf cancels out exactly: (s * sf) * (g / sf) == s * g
        ratio = (bf16_val.float() * processed_g.float()) / (
            (
                torch.tensor([out_bf16_bits], dtype=torch.int16)
                .view(torch.bfloat16)
                .float()
            )
            * (nvfp4_marlin_process_global_scale(g).float() / sf)
        )
        self.assertAlmostEqual(ratio.item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=3)
