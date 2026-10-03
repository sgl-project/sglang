"""Unit tests for _is_block_scale_fp8 per-channel vs block-scale fp8 detection.

Tests the helper that distinguishes block-scale fp8 (weight_scale [N, K/128],
compatible with fused gfx95 group-quant kernels) from per-channel fp8
(weight_scale [N, 1], must use the plain bf16 path).

These tests run on CPU and require no GPU, guarding the regression surface
cheaply without waiting for a full nightly accuracy run.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-a-test-1-gpu-small-amd")


def _make_proj(
    weight_dtype,
    weight_scale_shape=None,
    scale_name="weight_scale",
    scale_dtype=torch.float32,
):
    """Create a fake projection module with the given weight/scale configuration."""
    proj = SimpleNamespace()
    proj.weight = torch.empty(64, 512, dtype=weight_dtype)
    if weight_scale_shape is not None:
        scale = torch.empty(*weight_scale_shape, dtype=scale_dtype)
        setattr(proj, scale_name, scale)
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

    def test_block_scale_fp8_weight_scale_inv_returns_true(self):
        """Fp8LinearMethod block-scale: weight_scale_inv [N, K/128] — True.

        Native DeepSeek/GLM fp8 checkpoints ship the block scale as
        `weight_scale_inv` (45932 such tensors in DeepSeek-V3.2, zero named
        `weight_scale`), so reading only `weight_scale` misses them.
        """
        proj = _make_proj(
            torch.float8_e4m3fn,
            weight_scale_shape=(64, 4),
            scale_name="weight_scale_inv",
        )
        self.assertTrue(self.fn(proj))

    def test_per_channel_fp8_weight_scale_inv_returns_false(self):
        """Per-channel under the _inv name: weight_scale_inv [N, 1] — False."""
        proj = _make_proj(
            torch.float8_e4m3fn,
            weight_scale_shape=(64, 1),
            scale_name="weight_scale_inv",
        )
        self.assertFalse(self.fn(proj))

    def test_non_fp8_weight_with_weight_scale_inv_returns_false(self):
        """bf16 weight is not fp8 regardless of the scale name — False."""
        proj = _make_proj(
            torch.bfloat16,
            weight_scale_shape=(64, 4),
            scale_name="weight_scale_inv",
        )
        self.assertFalse(self.fn(proj))

    def test_mxfp8_group32_returns_false(self):
        """MXFP8: fp8 weight, 2-D uint8 weight_scale_inv, block [1, 32] — False.

        MXFP8 takes the same block_quant branch and also registers a 2-D
        `weight_scale_inv`, but K/32 scale columns and UE8M0 uint8 values are
        incompatible with the group-128 fused kernels, so a shape-only check
        would misroute it. K=512 with 16 columns implies block_k=32.
        """
        proj = _make_proj(
            torch.float8_e4m3fn,
            weight_scale_shape=(64, 16),
            scale_name="weight_scale_inv",
            scale_dtype=torch.uint8,
        )
        self.assertFalse(self.fn(proj))

    def test_non_128_block_k_returns_false(self):
        """Any block_k other than the fused group size is rejected (K/64 here)."""
        proj = _make_proj(
            torch.float8_e4m3fn,
            weight_scale_shape=(64, 8),
            scale_name="weight_scale_inv",
        )
        self.assertFalse(self.fn(proj))


class TestDetectGfx95QuantFormat(CustomTestCase):
    """`_detect_gfx95_quant_format` must find the scale under either name.

    It runs its own "has a scale been materialized yet?" probe before consulting
    `_is_block_scale_fp8`, so checking only `weight_scale` reported
    "fp8_pending" forever on a native DeepSeek/GLM fp8 checkpoint and left the
    layer permanently on the bf16 path.
    """

    def setUp(self):
        import sglang.srt.models.deepseek_v2 as dsv2

        self.dsv2 = dsv2
        if not dsv2._is_gfx95_supported:
            self.skipTest("requires gfx95x")
        self.fn = dsv2.DeepseekV2DecoderLayer._detect_gfx95_quant_format

    def _fake_layer(self, proj):
        return SimpleNamespace(
            self_attn=SimpleNamespace(fused_qkv_a_proj_with_mqa=proj)
        )

    def test_native_block_fp8_weight_scale_inv_detected_as_fp8(self):
        proj = _make_proj(
            torch.float8_e4m3fn,
            weight_scale_shape=(64, 4),
            scale_name="weight_scale_inv",
        )
        self.assertEqual(self.fn(self._fake_layer(proj)), "fp8")

    def test_per_channel_weight_scale_inv_detected_as_bf16(self):
        proj = _make_proj(
            torch.float8_e4m3fn,
            weight_scale_shape=(64, 1),
            scale_name="weight_scale_inv",
        )
        self.assertEqual(self.fn(self._fake_layer(proj)), "")

    def test_no_scale_yet_is_pending(self):
        proj = _make_proj(torch.float8_e4m3fn)
        self.assertEqual(self.fn(self._fake_layer(proj)), "fp8_pending")

    def test_mxfp4_uint8_weight_detected_as_mxfp4(self):
        proj = _make_proj(torch.uint8, weight_scale_shape=(64, 4))
        self.assertEqual(self.fn(self._fake_layer(proj)), "mxfp4")


if __name__ == "__main__":
    unittest.main()
