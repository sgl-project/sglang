"""CPU checks for saturating bf16 -> fp8 casts in ``_quantize_fp8_qkv``.

torch <= 2.12 (prod pins torch==2.11.0) implements ``Tensor.to(torch.float8)``
as NaN for every |x| >= 480 instead of saturating to the finite max (+-448 for
e4m3fn).  These tests patch ``torch.Tensor.to`` to emulate that legacy cast so
they fail on code that still uses the plain ``.to()`` and pass once the casts
go through ``to_fp8_satfinite``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import _quantize_fp8_qkv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn

_REAL_TO = torch.Tensor.to


def _legacy_fp8_to(self, *args, **kwargs):
    out = _REAL_TO(self, *args, **kwargs)
    if out.dtype == torch.float8_e4m3fn and self.dtype != torch.float8_e4m3fn:
        overflow = _REAL_TO(self, torch.float32).abs() >= 480.0
        out = torch.where(overflow, torch.full_like(out, float("nan")), out)
    return out


class TestQuantizeFp8Qkv(CustomTestCase):
    def test_bf16_qkv_with_overflow_produces_finite_fp8(self):
        layer = SimpleNamespace()  # no k/v scale -> scales default to 1.0
        shape = (4, 2, 8)
        q = torch.randn(shape, dtype=torch.bfloat16)
        k = torch.randn(shape, dtype=torch.bfloat16)
        v = torch.randn(shape, dtype=torch.bfloat16)
        q[0, 0, 0] = 1e4
        k[1, 0, 0] = 1e4
        v[2, 0, 0] = -1e4
        with patch.object(torch.Tensor, "to", _legacy_fp8_to):
            q_out, k_out, v_out, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)
        for t in (q_out, k_out, v_out):
            self.assertEqual(t.dtype, FP8)
            self.assertFalse(t.isnan().any().item())
        self.assertEqual((k_scale, v_scale), (1.0, 1.0))

    def test_prequantized_fp8_kv_returned_unchanged(self):
        layer = SimpleNamespace()
        q = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        k = torch.ones(4, 2, 8).to(FP8)
        v = torch.ones(4, 2, 4).to(FP8)
        with patch.object(torch.Tensor, "to", _legacy_fp8_to):
            _, k_out, v_out, _, _ = _quantize_fp8_qkv(q, k, v, layer)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)


if __name__ == "__main__":
    unittest.main()
