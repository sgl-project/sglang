"""gfx942 group32 ue8m0 block-FP8 linears served from the cached BF16 weight expansion."""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")

FP8_MAX = 448.0


def _quantize_block32_ue8m0(w: torch.Tensor, block: int = 32):
    """Checkpoint-format (w_fp8 [N, K], e8m0 scale [N/32, K/32]) and the dequant reference."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[:, None, :, None]).reshape(n, k)
    return w_fp8.reshape(n, k), scale, w_dequant


def _build_layer(n: int, k: int, weight=None, scale=None):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
    )
    layer = make_tp1_column_parallel_linear(
        quant_config, n, k, skip_block_quant_check=True
    )
    w_dequant = None
    if weight is None:
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        weight, scale, w_dequant = _quantize_block32_ue8m0(w)
    load_linear_weights(
        layer, weight=weight, weight_scale_inv=scale.to(torch.float8_e8m0fnu)
    )
    layer.quant_method.process_weights_after_loading(layer)
    return layer, w_dequant


@unittest.skipUnless(is_gfx942_supported(), "gfx942 BF16 expansion of block32 FP8")
class TestGfx942Block32Bf16Linear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def setUp(self):
        torch.manual_seed(7)

    def test_bf16_linear_matches_dequantized_weight(self):
        for n, k in ((576, 5120), (5120, 288), (96, 1280)):
            layer, w_dequant = _build_layer(n, k)
            cached = getattr(layer, "_block_fp8_bf16_weight", None)
            self.assertIsNotNone(cached)
            self.assertEqual(layer.weight.dtype, torch.float8_e4m3fnuz)
            torch.testing.assert_close(
                cached.float(), w_dequant.to(torch.bfloat16).float(), rtol=0, atol=0
            )
            bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            with mock.patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_triton",
                side_effect=AssertionError("gfx942 BF16 cache fell back to Triton"),
            ):
                for m in (1, 6, 65, 512):
                    with self.subTest(n=n, k=k, m=m):
                        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                        out = layer.quant_method.apply(layer, x)
                        ref = torch.nn.functional.linear(x, cached)
                        torch.testing.assert_close(out, ref, rtol=0, atol=0)
                        out = layer.quant_method.apply(layer, x, bias)
                        ref = torch.nn.functional.linear(x, cached, bias)
                        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_scale_range_follows_fnuz_normalization(self):
        # FNUZ normalization doubles the scale; the smallest E4M3 value times 2^-117
        # (checkpoint scale) is the smallest normal BF16.
        n, k = 32, 64
        weight = torch.ones((n, k), device="cuda").to(torch.float8_e4m3fn)
        for exp, cached in ((-117, True), (-118, False)):
            with self.subTest(exp=exp):
                scale = torch.full((n // 32, k // 32), 2.0**exp, device="cuda")
                layer, _ = _build_layer(n, k, weight.clone(), scale)
                has_cache = getattr(layer, "_block_fp8_bf16_weight", None) is not None
                self.assertEqual(has_cache, cached)

    def test_env_off_skips_cache(self):
        with envs.SGLANG_OPT_GFX942_BLOCK_FP8_BF16.override(False):
            layer, _ = _build_layer(64, 128)
        self.assertIsNone(getattr(layer, "_block_fp8_bf16_weight", None))
        self.assertIsNone(getattr(layer, "_derived_weight_cache_error", None))


if __name__ == "__main__":
    unittest.main()
