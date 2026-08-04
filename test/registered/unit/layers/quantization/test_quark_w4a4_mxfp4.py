"""Unit tests for QuarkW4A4MXFP4.apply_weights shape contract — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
)
from sglang.test.test_utils import CustomTestCase


def _make_layer(out_features: int, packed_in_half: int) -> SimpleNamespace:
    """Stub layer that exposes only the attributes apply_weights reads:
    layer.weight       (shape[0] = out_features)
    layer.weight_scale (passthrough; only identity-compared)
    """
    return SimpleNamespace(
        weight=torch.zeros(out_features, packed_in_half, dtype=torch.uint8),
        weight_scale=torch.zeros(out_features, packed_in_half // 16, dtype=torch.uint8),
    )


def _patch_aiter_kernels(out_features: int):
    """Patch the two aiter symbols referenced by `apply_weights` so the
    function runs on CPU without an actual GEMM.

    - `dynamic_mxfp4_quant(x)` returns a uint8 packed pair `(x_q, x_s)` whose
      first dim matches `x.shape[0]`. apply_weights only reads `x_q.shape[0]`
      and `x_q.device` from the result, so the exact contents don't matter.
    - `gemm_afp4wfp4(x_q, w, x_s, w_s, out_dtype, y)` is a no-op writer — the
      caller already allocated `y` at the correct shape, which is exactly what
      this test validates.

    `create=True` lets the patch succeed even when the module did not import
    these names (e.g. on a non-HIP CI host), so the same test exercises both
    environments.
    """
    target = "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4"

    def _fake_quant(x):
        n = x.shape[0]
        h_packed = max(1, x.shape[-1] // 2)
        x_q = torch.zeros(n, h_packed, dtype=torch.uint8, device=x.device)
        x_s = torch.zeros(n, max(1, h_packed // 16), dtype=torch.uint8, device=x.device)
        return x_q, x_s

    return (
        patch(f"{target}.dynamic_mxfp4_quant", side_effect=_fake_quant, create=True),
        patch(f"{target}.gemm_afp4wfp4", return_value=None, create=True),
    )


class TestQuarkW4A4MXFP4ApplyWeightsShape(CustomTestCase):
    """Regression tests for the 3-D output_shape bug.

    Bug: output_shape = [*x.shape[:-1], layer.weight.shape[0]] was computed
    AFTER `x = x.view(-1, x.shape[-1])`, so x.shape[:-1] was already
    [B*S] instead of [B, S]. The function then returned a 2-D tensor where
    callers expected 3-D, silently corrupting the shape contract.
    """

    def setUp(self):
        # The two specs are stored but never read on the path under test.
        self.scheme = QuarkW4A4MXFP4(weight_quant_spec={}, input_quant_spec={})
        # `out_dtype` is captured at __init__ from torch.get_default_dtype();
        # pin it so the empty(y) allocation is deterministic.
        self.scheme.out_dtype = torch.bfloat16

    # ---- Bug-catchers: must FAIL on the unfixed code ------------------------

    def test_3d_input_returns_3d_output(self):
        # Bug facet: 3-D activation in → 3-D activation out, with the original
        # batch/seq dims preserved (not collapsed to B*S).
        B, S, H, OUT = 2, 4, 64, 128
        x = torch.randn(B, S, H, dtype=torch.bfloat16)
        layer = _make_layer(out_features=OUT, packed_in_half=H // 2)

        quant_patch, gemm_patch = _patch_aiter_kernels(OUT)
        with quant_patch, gemm_patch:
            out = self.scheme.apply_weights(layer, x)

        self.assertEqual(
            out.dim(), 3, f"expected 3-D output, got shape {tuple(out.shape)}"
        )
        self.assertEqual(tuple(out.shape), (B, S, OUT))

    def test_3d_input_preserves_batch_dim_separately(self):
        # Extra coverage: vary B and S to confirm both dims are preserved
        # individually (and not, e.g., B and S getting swapped).
        for B, S in [(1, 8), (3, 1), (5, 7)]:
            with self.subTest(B=B, S=S):
                H, OUT = 32, 16
                x = torch.randn(B, S, H, dtype=torch.bfloat16)
                layer = _make_layer(out_features=OUT, packed_in_half=H // 2)

                quant_patch, gemm_patch = _patch_aiter_kernels(OUT)
                with quant_patch, gemm_patch:
                    out = self.scheme.apply_weights(layer, x)

                self.assertEqual(tuple(out.shape), (B, S, OUT))

    # ---- Guardrails: must pass on both buggy and fixed code -----------------

    def test_2d_input_returns_2d_output(self):
        # 2-D path is unchanged by the fix — keep it as a regression guard.
        N, H, OUT = 6, 64, 128
        x = torch.randn(N, H, dtype=torch.bfloat16)
        layer = _make_layer(out_features=OUT, packed_in_half=H // 2)

        quant_patch, gemm_patch = _patch_aiter_kernels(OUT)
        with quant_patch, gemm_patch:
            out = self.scheme.apply_weights(layer, x)

        self.assertEqual(tuple(out.shape), (N, OUT))


if __name__ == "__main__":
    unittest.main()
