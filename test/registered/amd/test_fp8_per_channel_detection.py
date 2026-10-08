"""Unit tests for per-channel vs block-scale fp8 detection.

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

register_amd_ci(est_time=10, suite="stage-a-test-1-gpu-small-amd")


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


class TestAcceptsGroup128Fp8Tuple(unittest.TestCase):
    """Unit tests for the ROCm accepts_group128_fp8_tuple probe.

    It answers a different question than _is_block_scale_fp8: "may a producer
    kernel hand this linear (fp8, group-128 scale) instead of bf16?". So it also
    accepts the fnuz dtype, the weight_scale_inv spelling, and a quant_method
    that declares [128, 128] blocks before the scale tensor is materialized.
    """

    def setUp(self):
        from sglang.srt.models.deepseek_common.utils_rocm import (
            accepts_group128_fp8_tuple,
        )

        self.fn = accepts_group128_fp8_tuple

    def test_fnuz_weight_accepted(self):
        """gfx942 stores fp8 as e4m3fnuz; the fn dtype list alone misses it."""
        proj = _make_proj(torch.float8_e4m3fnuz, weight_scale_shape=(64, 4))
        self.assertTrue(self.fn(proj))

    def test_weight_scale_inv_accepted(self):
        """Block-scale checkpoints name the scale weight_scale_inv."""
        proj = _make_proj(torch.float8_e4m3fn)
        proj.weight_scale_inv = torch.empty(64, 4, dtype=torch.float32)
        self.assertTrue(self.fn(proj))

    def test_quant_method_block_size_accepted(self):
        """No scale tensor yet, but quant_method declares [128, 128] blocks."""
        proj = _make_proj(torch.float8_e4m3fn)
        proj.quant_method = SimpleNamespace(weight_block_size=[128, 128])
        self.assertTrue(self.fn(proj))

    def test_per_channel_fp8_returns_false(self):
        """Per-channel fp8 [N, 1] needs the PTPC probe, not this one."""
        proj = _make_proj(torch.float8_e4m3fn, weight_scale_shape=(64, 1))
        proj.quant_method = SimpleNamespace(weight_block_size=None)
        self.assertFalse(self.fn(proj))

    def test_non_fp8_weight_returns_false(self):
        """bf16 weight is not fp8 at all, whatever quant_method claims."""
        proj = _make_proj(torch.bfloat16, weight_scale_shape=(64, 4))
        proj.quant_method = SimpleNamespace(weight_block_size=[128, 128])
        self.assertFalse(self.fn(proj))


if __name__ == "__main__":
    unittest.main()
