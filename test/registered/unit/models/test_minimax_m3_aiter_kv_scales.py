"""The FP8 mode choice for MiniMax-M3's aiter fused qk-norm + rope + KV-store path.

aiter has two FP8 cache modes and they are NOT interchangeable
(csrc/kernels/fused_qknorm_idxrqknorm.cu asserts both contracts):

* ``fp8_e4m3_unit``   -- scale fixed at 1.0, and the kernel REFUSES k/v scale
  tensors ("fp8_e4m3_unit does not accept k_scale/v_scale tensors").
* ``fp8_e4m3_static`` -- the kernel REQUIRES single-element contiguous fp32
  scale tensors.

So the mode has to be chosen from the pool's actual per-tensor scales. Choosing
unit unconditionally writes a non-unit scale to the cache unscaled, which is a
silent numerical error rather than a crash -- exactly the regression this file
exists to catch.

``_aiter_fused_kv_scales`` is a pure predicate over a small attribute surface, so
it is driven here with a stub rather than a real attention module: no accelerator
beyond the import, and every branch gets its own case.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import minimax_m3
from sglang.srt.models.minimax_m3 import MiniMaxM3Attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KV_SCALES = MiniMaxM3Attention._aiter_fused_kv_scales


def _layer(k_scale, v_scale, cached=None):
    """The four attributes _aiter_fused_kv_scales reaches for."""
    return SimpleNamespace(
        attn=SimpleNamespace(k_scale_float=k_scale, v_scale_float=v_scale),
        _aiter_kv_scales=cached,
        q_norm=SimpleNamespace(weight=torch.zeros(1, device="cpu")),
    )


class TestAiterFusedKvScales(CustomTestCase):
    def test_absent_scales_select_the_unit_mode(self):
        """None means unit by SGLang's convention, and unit must pass NO tensors."""
        self.assertEqual(_KV_SCALES(_layer(None, None)), ("fp8_e4m3_unit", None, None))

    def test_explicit_unit_scales_select_the_unit_mode(self):
        self.assertEqual(_KV_SCALES(_layer(1.0, 1.0)), ("fp8_e4m3_unit", None, None))

    def test_mixed_absent_and_unit_scales_select_the_unit_mode(self):
        self.assertEqual(_KV_SCALES(_layer(None, 1.0)), ("fp8_e4m3_unit", None, None))
        self.assertEqual(_KV_SCALES(_layer(1.0, None)), ("fp8_e4m3_unit", None, None))

    def test_non_unit_k_scale_selects_the_static_mode(self):
        """A single non-unit scale is enough: unit would write it unscaled."""
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=False):
            mode, k, v = _KV_SCALES(_layer(0.5, 1.0))
        self.assertEqual(mode, "fp8_e4m3_static")
        self.assertEqual(k.item(), 0.5)
        self.assertEqual(v.item(), 1.0)

    def test_non_unit_v_scale_selects_the_static_mode(self):
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=False):
            mode, k, v = _KV_SCALES(_layer(1.0, 0.25))
        self.assertEqual(mode, "fp8_e4m3_static")
        self.assertEqual((k.item(), v.item()), (1.0, 0.25))

    def test_absent_counterpart_defaults_to_unit_not_none(self):
        """None alongside a non-unit scale must become 1.0, never a null tensor."""
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=False):
            _, k, v = _KV_SCALES(_layer(None, 0.5))
        self.assertEqual((k.item(), v.item()), (1.0, 0.5))

    def test_static_scales_satisfy_the_kernel_contract(self):
        """Single-element, contiguous, fp32 -- what the kernel asserts it gets."""
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=False):
            _, k, v = _KV_SCALES(_layer(0.5, 0.25))
        for t in (k, v):
            self.assertEqual(t.numel(), 1)
            self.assertTrue(t.is_contiguous())
            self.assertEqual(t.dtype, torch.float32)

    def test_scales_are_cached_on_the_layer(self):
        """Built once: allocating per call would add a launch to the path."""
        layer = _layer(0.5, 0.25)
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=False):
            _, k1, v1 = _KV_SCALES(layer)
            _, k2, v2 = _KV_SCALES(layer)
        self.assertIs(k1, k2)
        self.assertIs(v1, v2)

    def test_cached_scales_are_reused_without_allocating(self):
        cached = (torch.tensor([0.5]), torch.tensor([0.25]))
        layer = _layer(0.5, 0.25, cached=cached)
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=True):
            mode, k, v = _KV_SCALES(layer)
        self.assertEqual(mode, "fp8_e4m3_static")
        self.assertIs(k, cached[0])

    def test_capture_before_warmup_falls_back_rather_than_allocating(self):
        """Allocating under graph capture is not allowed, so the path declines."""
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=True):
            self.assertIsNone(_KV_SCALES(_layer(0.5, 0.25)))

    def test_unit_mode_does_not_need_capture_mode_at_all(self):
        """The unit path allocates nothing, so it stays open during capture."""
        with patch.object(minimax_m3, "get_is_capture_mode", return_value=True):
            self.assertEqual(
                _KV_SCALES(_layer(None, None)), ("fp8_e4m3_unit", None, None)
            )


if __name__ == "__main__":
    unittest.main()
