"""gfx942 group32 ue8m0 block-FP8 linears served from the cached BF16 weight expansion."""

import contextlib
import unittest
from unittest import mock

import torch

from sglang.srt.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
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
        # FNUZ normalization doubles the scale, so checkpoint scales keep the E4M3
        # exactness range [2^-117, 2^119].
        n, k = 32, 64
        weight = torch.ones((n, k), device="cuda").to(torch.float8_e4m3fn)
        for exp, cached in ((-117, True), (-118, False), (119, True), (120, False)):
            with self.subTest(exp=exp):
                scale = torch.full((n // 32, k // 32), 2.0**exp, device="cuda")
                layer, _ = _build_layer(n, k, weight.clone(), scale)
                has_cache = getattr(layer, "_block_fp8_bf16_weight", None) is not None
                self.assertEqual(has_cache, cached)
                # The rejection depends only on the config, so a shard whose scales
                # keep FP8 rejects a reload before writing, like its cached peers.
                before = layer.weight.detach().view(torch.uint8).clone()
                updater = _weight_updater(torch.nn.Sequential(layer))
                ok, error = updater.update_weights_from_disk("/nonexistent", "auto")
                self.assertFalse(ok)
                self.assertIn("gfx942", error)
                self.assertIn("gfx942", updater.load_rejection("plugin_weight_update"))
                self.assertTrue(torch.equal(layer.weight.view(torch.uint8), before))

    def test_cache_rejects_online_updates(self):
        layer, _ = _build_layer(64, 128)
        self.assertIsNotNone(getattr(layer, "_block_fp8_bf16_weight", None))
        self.assertIn("gfx942", layer._derived_weight_cache_error)

    def test_batch_invariant_mode_keeps_cache(self):
        # A draft model loads after the target has enabled batch-invariant mode.
        with set_batch_invariant_mode(True):
            layer, _ = _build_layer(576, 5120)
            bias = torch.randn(576, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(135, 5120, device="cuda", dtype=torch.bfloat16)
            full = layer.quant_method.apply(layer, x, bias)
            for m in (1, 6, 65):
                with self.subTest(m=m):
                    out = layer.quant_method.apply(layer, x[:m], bias)
                    torch.testing.assert_close(out, full[:m], rtol=0, atol=0)
        self.assertIsNotNone(getattr(layer, "_block_fp8_bf16_weight", None))
        self.assertIn("gfx942", layer._derived_weight_cache_error)

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
        self.assertIn("gfx942", layer._derived_weight_cache_error)
        self.assertTrue(fake_matmul.called)


def _weight_updater(model: torch.nn.Module) -> WeightUpdater:
    return WeightUpdater(
        tp_rank=0,
        device="cuda",
        gpu_id=0,
        model_config=mock.Mock(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=mock.Mock(),
        recapture_cuda_graph=mock.Mock(),
        get_model_runner=mock.Mock(),
    )


@contextlib.contextmanager
def _fake_fp8_kernels(n: int):
    """Stand-ins for the FP8 Triton kernels; only the dispatch is checked."""

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
