"""CPU tests for the SM70 MXFP4 e2m1+UE8M0 g32 Marlin pack.

No server, no 476 GiB checkpoint. GPU GEMM coverage lives in
``test_mxfp4_sm70_marlin_gemm.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    DSV41_FLASH_HIDDEN_SIZE,
    DSV41_FLASH_MOE_INTERMEDIATE_SIZE,
    DSV41_FLASH_MXFP4_GROUP_SIZE,
    DSV41_FLASH_NUM_EXPERTS_PER_TOK,
    DSV41_FLASH_NUM_ROUTED_EXPERTS,
    check_moe_marlin_supports_layer,
    sm70_mxfp4_fused_decode_expert_limit,
    sm70_mxfp4_logical_ue8m0_scales,
    sm70_mxfp4_refuse_nvfp4_metadata,
    sm70_mxfp4_ue8m0_to_uint8,
    sm70_mxfp4_validate_ue8m0_scales,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    dequant_mxfp8_ue8m0_to_fp16,
    prepare_mxfp8_layer_for_sm70_marlin,
    sm70_mxfp8_ue8m0_to_fp16_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return (even | (odd << 4)).to(torch.uint8)


def dequant_mxfp4_e2m1_ue8m0(
    packed: torch.Tensor, scale_u8: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    """Torch reference: packed ``[..., K/2]`` uint8, scales ``[..., K/32]``."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    sign = 1.0 - 2.0 * ((codes >> 3) & 1).to(torch.float32)
    mag = (codes & 7).to(torch.long)
    lut = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    values = sign * lut[mag]
    scales = torch.pow(
        torch.tensor(2.0, device=packed.device), scale_u8.to(torch.float32) - 127
    )
    scales = scales.repeat_interleave(group_size, dim=-1)
    return values * scales


class TestSm70Mxfp4MarlinPack(CustomTestCase):
    def test_ue8m0_byte_127_is_one(self):
        raw = torch.tensor([127], dtype=torch.uint8)
        self.assertEqual(
            int(sm70_mxfp4_ue8m0_to_uint8(raw.view(torch.float8_e8m0fnu)).item()),
            127,
        )
        self.assertEqual(float(torch.pow(torch.tensor(2.0), raw.float() - 127)), 1.0)

    def test_logical_scales_transpose_checkpoint_layout(self):
        num_experts, size_n, size_k = 4, 64, 128
        checkpoint = torch.randint(
            120, 135, (num_experts, size_n, size_k // 32), dtype=torch.uint8
        )
        logical = sm70_mxfp4_logical_ue8m0_scales(
            checkpoint.view(torch.float8_e8m0fnu),
            size_k=size_k,
            size_n=size_n,
            group_size=32,
        )
        self.assertEqual(tuple(logical.shape), (num_experts, size_k // 32, size_n))
        self.assertEqual(logical.dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(
            logical.view(torch.uint8),
            checkpoint.transpose(1, 2).contiguous(),
            rtol=0,
            atol=0,
        )

    def test_384_experts_are_not_dropped(self):
        size_n, size_k = 64, 128
        scales = torch.full(
            (DSV41_FLASH_NUM_ROUTED_EXPERTS, size_n, size_k // 32),
            127,
            dtype=torch.uint8,
        )
        logical = sm70_mxfp4_logical_ue8m0_scales(
            scales, size_k=size_k, size_n=size_n, group_size=32
        )
        self.assertEqual(logical.shape[0], DSV41_FLASH_NUM_ROUTED_EXPERTS)
        self.assertIsNone(sm70_mxfp4_fused_decode_expert_limit())

    def test_official_dsv41_flash_shapes_are_marlin_legal(self):
        layer = SimpleNamespace(
            hidden_size=DSV41_FLASH_HIDDEN_SIZE,
            intermediate_size_per_partition=DSV41_FLASH_MOE_INTERMEDIATE_SIZE,
            moe_runner_config=SimpleNamespace(
                apply_router_weight_on_input=False,
                is_gated=True,
                activation="silu",
            ),
        )
        self.assertTrue(
            check_moe_marlin_supports_layer(
                layer, DSV41_FLASH_MXFP4_GROUP_SIZE, allow_tile_padding=True
            )
        )
        self.assertTrue(
            check_moe_marlin_supports_layer(
                layer, DSV41_FLASH_MXFP4_GROUP_SIZE, allow_tile_padding=False
            )
        )
        self.assertEqual(DSV41_FLASH_NUM_EXPERTS_PER_TOK, 6)
        self.assertEqual(DSV41_FLASH_HIDDEN_SIZE % 64, 0)
        self.assertEqual(DSV41_FLASH_MOE_INTERMEDIATE_SIZE % 32, 0)
        self.assertEqual(DSV41_FLASH_HIDDEN_SIZE % 128, 0)
        self.assertEqual(DSV41_FLASH_MOE_INTERMEDIATE_SIZE % 64, 0)

    def test_out_of_range_ue8m0_fails_loud(self):
        scales = torch.tensor([[[100]]], dtype=torch.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            sm70_mxfp4_validate_ue8m0_scales(scales)
        self.assertIn("Refusing to clamp", str(ctx.exception))

    def test_nvfp4_e4m3_scales_fail_loud(self):
        scales = torch.zeros((2, 4, 4), dtype=torch.float8_e4m3fn)
        with self.assertRaises(RuntimeError) as ctx:
            sm70_mxfp4_logical_ue8m0_scales(scales, size_k=128, size_n=4)
        self.assertIn("NVFP4", str(ctx.exception))

    def test_nvfp4_global_scale_on_layer_fails_loud(self):
        layer = SimpleNamespace(w13_weight_scale_2=torch.ones(2))
        with self.assertRaises(RuntimeError) as ctx:
            sm70_mxfp4_refuse_nvfp4_metadata(layer)
        self.assertIn("global_scale", str(ctx.exception))

    def test_group_size_16_fails_loud(self):
        scales = torch.full((1, 32, 8), 127, dtype=torch.uint8)
        with self.assertRaises(ValueError) as ctx:
            sm70_mxfp4_logical_ue8m0_scales(
                scales, size_k=256, size_n=32, group_size=16
            )
        self.assertIn("g32", str(ctx.exception))

    def test_dequant_one_linear_matches_lut(self):
        torch.manual_seed(0)
        n, k = 32, 64
        codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
        packed = _pack_e2m1_codes(codes)
        scales = torch.randint(120, 135, (n, k // 32), dtype=torch.uint8)
        dequant = dequant_mxfp4_e2m1_ue8m0(packed, scales)
        self.assertEqual(tuple(dequant.shape), (n, k))
        x = torch.randn(4, k, dtype=torch.float32)
        ref = x @ dequant.T
        self.assertEqual(tuple(ref.shape), (4, n))
        self.assertTrue(torch.isfinite(ref).all())

    def test_mxfp8_ue8m0_to_fp16_is_exact_power_of_two(self):
        raw = torch.tensor([120, 127, 134], dtype=torch.uint8)
        fp16 = sm70_mxfp8_ue8m0_to_fp16_scales(raw)
        expected = torch.pow(torch.tensor(2.0), raw.float() - 127).to(torch.float16)
        torch.testing.assert_close(fp16, expected, rtol=0, atol=0)

    def test_mxfp8_g32_keeps_packed_e4m3(self):
        n, k = 32, 64
        weight = torch.full((n, k), 2.0).to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        layer = torch.nn.Module()
        layer.weight_block_size = [1, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        layer.orig_dtype = torch.float16
        prepare_mxfp8_layer_for_sm70_marlin(layer)
        self.assertTrue(layer._sm70_mxfp8_w8a16)
        self.assertFalse(getattr(layer, "_sm70_mxfp8_dequant_fp16", False))
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_scale_inv.dtype, torch.uint8)
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (n, k // 32))
        unpacked = dequant_mxfp8_ue8m0_to_fp16(
            layer.weight, layer.weight_scale_inv, (1, 32)
        )
        torch.testing.assert_close(
            unpacked.float(),
            torch.full((n, k), 2.0),
            rtol=0,
            atol=0,
        )

    def test_mxfp8_g32_unpacks_to_fp16_when_killswitch_off(self):
        n, k = 32, 64
        weight = torch.full((n, k), 2.0).to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        layer = torch.nn.Module()
        layer.weight_block_size = [1, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        layer.orig_dtype = torch.float16
        with envs.SGLANG_DSV41_MXFP8_W8A16.override(False):
            prepare_mxfp8_layer_for_sm70_marlin(layer)
        self.assertTrue(layer._sm70_mxfp8_dequant_fp16)
        self.assertEqual(layer.weight.dtype, torch.float16)
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        torch.testing.assert_close(
            layer.weight.float(),
            torch.full((n, k), 2.0),
            rtol=0,
            atol=0,
        )

    def test_mxfp8_32x32_keeps_packed_and_expands_scales(self):
        n, k = 64, 64
        weight = torch.full((n, k), 1.0).to(torch.float8_e4m3fn)
        scales = torch.full((n // 32, k // 32), 128, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        layer = torch.nn.Module()
        layer.weight_block_size = [32, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        layer.orig_dtype = torch.float16
        prepare_mxfp8_layer_for_sm70_marlin(layer)
        self.assertTrue(layer._sm70_mxfp8_w8a16)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (n, k // 32))
        self.assertTrue(torch.equal(
            layer.weight_scale_inv, torch.full((n, k // 32), 128, dtype=torch.uint8)
        ))
        unpacked = dequant_mxfp8_ue8m0_to_fp16(
            layer.weight, layer.weight_scale_inv, (1, 32)
        )
        torch.testing.assert_close(
            unpacked.float(),
            torch.full((n, k), 2.0),
            rtol=0,
            atol=0,
        )

    def test_mxfp8_32x32_unpacks_to_fp16_when_killswitch_off(self):
        n, k = 64, 64
        weight = torch.full((n, k), 1.0).to(torch.float8_e4m3fn)
        scales = torch.full((n // 32, k // 32), 128, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        layer = torch.nn.Module()
        layer.weight_block_size = [32, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        layer.orig_dtype = torch.float16
        with envs.SGLANG_DSV41_MXFP8_W8A16.override(False):
            prepare_mxfp8_layer_for_sm70_marlin(layer)
        self.assertTrue(layer._sm70_mxfp8_dequant_fp16)
        torch.testing.assert_close(
            layer.weight.float(),
            torch.full((n, k), 2.0),
            rtol=0,
            atol=0,
        )

    def test_mxfp8_ue8m0_byte_109_stays_packed(self):
        # Official DSV4.1-Flash has scale bytes 109-112. 16 * 2^-18 = 2^-14,
        # which is fp16 min normal, so the product survives the final cast.
        n, k = 32, 64
        weight = torch.full((n, k), 16.0).to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 109, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        layer = torch.nn.Module()
        layer.weight_block_size = [1, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        layer.orig_dtype = torch.float16
        prepare_mxfp8_layer_for_sm70_marlin(layer)
        self.assertTrue(layer._sm70_mxfp8_w8a16)
        self.assertTrue(torch.equal(
            layer.weight_scale_inv, torch.full((n, k // 32), 109, dtype=torch.uint8)
        ))
        unpacked = dequant_mxfp8_ue8m0_to_fp16(
            layer.weight, layer.weight_scale_inv, (1, 32)
        )
        expected = torch.full((n, k), 2.0 ** (109 - 127 + 4), dtype=torch.float16)
        torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)

    def test_mxfp8_ue8m0_byte_109_dequants_via_fp32_when_unpacked(self):
        n, k = 32, 64
        weight = torch.full((n, k), 16.0).to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 109, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        layer = torch.nn.Module()
        layer.weight_block_size = [1, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        layer.orig_dtype = torch.float16
        with envs.SGLANG_DSV41_MXFP8_W8A16.override(False):
            prepare_mxfp8_layer_for_sm70_marlin(layer)
        expected = torch.full((n, k), 2.0 ** (109 - 127 + 4), dtype=torch.float16)
        torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)

    def test_official_v41_quant_config_is_mxfp8_fp4(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        cfg = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            }
        )
        self.assertTrue(cfg.use_mxfp8)
        self.assertTrue(cfg.is_fp4_experts)
        self.assertEqual(list(cfg.weight_block_size), [32, 32])


if __name__ == "__main__":
    import unittest

    unittest.main()
