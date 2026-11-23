"""gfx942 group32 ue8m0 block-FP8 linears served from the cached BF16 weight expansion."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8, fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="base-b-test-1-gpu-small-amd")

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
                # An all-FP8 layer keeps online updates.
                rejects = getattr(layer, "_derived_weight_cache_error", None)
                self.assertEqual(rejects is not None, cached)

    def test_cache_rejects_online_updates(self):
        layer, _ = _build_layer(64, 128)
        self.assertIsNotNone(getattr(layer, "_block_fp8_bf16_weight", None))
        self.assertIn("gfx942", layer._derived_weight_cache_error)

    def test_cache_follows_every_tp_rank(self):
        # A peer whose shard fails the scale check keeps this rank on FP8 too, so the
        # online-update rejection agrees across ranks.
        tp_group = SimpleNamespace(world_size=2, cpu_group=object())

        def peer_fails(flags, op=None, group=None):
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            self.assertIs(group, tp_group.cpu_group)
            flags.zero_()

        with (
            mock.patch.object(
                fp8, "get_parallel", lambda: SimpleNamespace(tp_group=tp_group)
            ),
            mock.patch.object(
                torch.distributed, "all_reduce", side_effect=peer_fails
            ) as all_reduce,
        ):
            layer, _ = _build_layer(64, 128)
        self.assertEqual(all_reduce.call_count, 1)
        self.assertIsNone(getattr(layer, "_block_fp8_bf16_weight", None))
        self.assertIsNone(getattr(layer, "_derived_weight_cache_error", None))

    def test_explicit_triton_backend_keeps_fp8_path(self):
        with (
            mock.patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                fp8_utils.Fp8GemmRunnerBackend.TRITON,
            ),
            _fake_fp8_kernels(64) as fake_matmul,
        ):
            layer, _ = _build_layer(64, 128)
            layer.quant_method.apply(
                layer, torch.randn(6, 128, device="cuda").bfloat16()
            )
        self.assertIsNone(getattr(layer, "_block_fp8_bf16_weight", None))
        self.assertIsNone(getattr(layer, "_derived_weight_cache_error", None))
        self.assertTrue(fake_matmul.called)


@contextlib.contextmanager
def _fake_fp8_kernels(n: int):
    """Stand-ins for the FP8 Triton path, whose ROCm group32 ue8m0 activation quant
    fails on its own; only the dispatch is checked."""

    def fake_group_quant(x, group_size, **kwargs):
        q = torch.zeros_like(x, dtype=torch.float8_e4m3fnuz)
        return q, torch.ones(x.shape[0], x.shape[1] // group_size, device=x.device)

    def fake_matmul(q_input, *args, **kwargs):
        return torch.zeros(q_input.shape[0], n, device=q_input.device)

    matmul = mock.Mock(side_effect=fake_matmul)
    with (
        mock.patch.object(
            fp8_utils, "sglang_per_token_group_quant_fp8", fake_group_quant
        ),
        mock.patch.object(fp8_utils, "per_token_group_quant_fp8", fake_group_quant),
        mock.patch.object(fp8_utils, "w8a8_block_fp8_matmul_triton", matmul),
    ):
        yield matmul


if __name__ == "__main__":
    unittest.main()
