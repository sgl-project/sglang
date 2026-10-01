"""CPU unit tests for AMD fp8 weight-scale handling (no GPU required).

Two independent regression surfaces, guarded cheaply without a nightly run:

- ``_is_block_scale_fp8``: distinguishes block-scale fp8 (weight_scale [N, K/128],
  compatible with fused gfx95 group-quant kernels) from per-channel fp8
  (weight_scale [N, 1], must use the plain bf16 path).
- ``Fp8LinearMethod.process_weights_after_loading`` on the e4m3fnuz aiter
  per-token path: a static activation input_scale must be doubled alongside
  weight_scale (see TestFnuzAiterPerTokenInputScale).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-a-test-1-gpu-small-amd")


def _make_proj(weight_dtype, weight_scale_shape=None):
    """Create a fake projection module with the given weight/scale configuration."""
    proj = SimpleNamespace()
    proj.weight = torch.empty(64, 512, dtype=weight_dtype)
    if weight_scale_shape is not None:
        proj.weight_scale = torch.empty(*weight_scale_shape, dtype=torch.float32)
    return proj


class TestIsBlockScaleFp8(unittest.TestCase):
    """Unit tests for _is_block_scale_fp8 detection helper."""

    def setUp(self):
        from sglang.srt.models.deepseek_common.utils import _is_block_scale_fp8

        self.fn = _is_block_scale_fp8

    def test_block_scale_fp8_returns_true(self):
        """Block-scale fp8: weight_scale [N, K/128] — should return True."""
        proj = _make_proj(torch.float8_e4m3fn, weight_scale_shape=(64, 4))
        self.assertTrue(self.fn(proj))

    def test_per_channel_fp8_returns_false(self):
        """Per-channel fp8: weight_scale [N, 1] — should return False."""
        proj = _make_proj(torch.float8_e4m3fn, weight_scale_shape=(64, 1))
        self.assertFalse(self.fn(proj))

    def test_non_fp8_weight_returns_false(self):
        """bf16 weight is not fp8 at all — should return False."""
        proj = _make_proj(torch.bfloat16, weight_scale_shape=(64, 4))
        self.assertFalse(self.fn(proj))

    def test_uint8_mxfp4_returns_false(self):
        """uint8 mxfp4 weight — should return False (handled separately)."""
        proj = _make_proj(torch.uint8, weight_scale_shape=(64, 4))
        self.assertFalse(self.fn(proj))

    def test_no_weight_scale_returns_false(self):
        """No weight_scale attribute — should return False gracefully."""
        proj = _make_proj(torch.float8_e4m3fn)  # no weight_scale
        self.assertFalse(self.fn(proj))

    def test_1d_weight_scale_returns_false(self):
        """1D weight_scale [N] (not yet reshaped) — should return False."""
        proj = _make_proj(torch.float8_e4m3fn, weight_scale_shape=(64,))
        self.assertFalse(self.fn(proj))

    def test_no_weight_attribute_returns_false(self):
        """No weight attribute — should return False gracefully."""
        proj = SimpleNamespace()
        self.assertFalse(self.fn(proj))


def _process_with_fnuz_aiter_per_token(activation_scheme, input_scale_value):
    """Drive the fnuz + aiter-per-token weight-processing path and return the
    resulting ``layer.input_scale`` (the value apply_fp8_linear would consume)."""
    from sglang.srt.layers.quantization import fp8

    method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
    method.block_quant = False
    method.use_mxfp8 = False
    method.cutlass_fp8_supported = False
    method.use_marlin = False
    method.is_checkpoint_fp8_serialized = True
    method.use_aiter_fp8_per_token = True
    method.use_per_token_if_dynamic = False
    method.quant_config = SimpleNamespace(
        activation_scheme=activation_scheme, weight_block_size=None
    )

    n, k = 32, 64
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(n, k).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor([0.5], dtype=torch.float32), requires_grad=False
    )
    if input_scale_value is None:
        layer.input_scale = None
    else:
        layer.input_scale = torch.nn.Parameter(
            torch.tensor([input_scale_value], dtype=torch.float32), requires_grad=False
        )
    layer.logical_widths = [n]

    # Force the fnuz + aiter per-token branch deterministically regardless of the
    # runner's arch/env, and neutralize the real aiter weight shuffle.
    with (
        mock.patch.object(fp8, "_is_fp8_fnuz", True),
        mock.patch.object(fp8, "_use_aiter", True),
        mock.patch.object(fp8, "_is_cpu", False),
        mock.patch.object(fp8, "use_aiter_bpreshuffle_gemm", return_value=False),
    ):
        method.process_weights_after_loading(layer)

    return layer


class TestFnuzAiterPerTokenInputScale(unittest.TestCase):
    """Regression: on ROCm e4m3fnuz, Fp8LinearMethod.process_weights_after_loading
    doubled weight_scale but not a static activation input_scale on the
    ``_use_aiter and use_aiter_fp8_per_token`` branch. A static input_scale is
    still consumed by apply_fp8_linear, so leaving it un-doubled quantizes
    activations against a 2x-overrange scale -> wrong GEMM results.
    """

    def test_static_input_scale_is_doubled(self):
        """Static activation scale must be doubled to match the fnuz weight scale."""
        layer = _process_with_fnuz_aiter_per_token(
            activation_scheme="static", input_scale_value=0.5
        )
        # weight_scale doubling is the existing behavior that defines the target;
        # input_scale must track it (0.5 -> 1.0). Pre-fix it stayed 0.5.
        self.assertAlmostEqual(layer.weight_scale.flatten()[0].item(), 1.0, places=6)
        self.assertAlmostEqual(layer.input_scale.item(), 1.0, places=6)

    def test_dynamic_input_scale_stays_none(self):
        """Dynamic activation (input_scale is None) must not crash or be set."""
        layer = _process_with_fnuz_aiter_per_token(
            activation_scheme="dynamic", input_scale_value=None
        )
        self.assertIsNone(layer.input_scale)


if __name__ == "__main__":
    unittest.main()
