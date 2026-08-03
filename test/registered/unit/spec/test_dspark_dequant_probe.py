"""Hermetic tests for the DSPARK stacked-projection support probes.

The probes decide whether the fused KV-projection stacking can read a
draft linear's weight; their contract is to *answer*, never to raise —
unsupported schemes must route to the per-linear torch fallback. A
W4A16 checkpoint exposed the gap: packed quant methods store ``qweight``
and have no ``weight`` attribute, so the probe's own attribute access
raised and killed the scheduler on the first speculative prefill. Pure
CPU, no weights, no GPU.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    _dequant_supported,
    _fused_commit_kv_proj_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _FakeLinear(torch.nn.Module):
    """Bare module: attributes exist only if handed in — like a quantized
    ReplicatedLinear, missing ``weight`` raises AttributeError on access."""

    def __init__(self, **attrs):
        super().__init__()
        self.quant_method = object()  # no block_quant attribute
        for k, v in attrs.items():
            setattr(self, k, v)


class TestDequantProbe(CustomTestCase):
    def test_packed_qweight_only_answers_false(self):
        m = _FakeLinear(qweight=torch.zeros(8, 8, dtype=torch.int32))
        self.assertFalse(_dequant_supported(m))

    def test_plain_bf16_weight_answers_true(self):
        m = _FakeLinear(weight=torch.zeros(8, 8, dtype=torch.bfloat16))
        self.assertTrue(_dequant_supported(m))

    def test_fp8_with_matching_scale_answers_true(self):
        m = _FakeLinear(
            weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(2, 2),
        )
        self.assertTrue(_dequant_supported(m))

    def test_fp8_missing_scale_answers_false(self):
        m = _FakeLinear(weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn))
        self.assertFalse(_dequant_supported(m))

    def test_fp8_wrong_scale_shape_answers_false(self):
        m = _FakeLinear(
            weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(3, 3),
        )
        self.assertFalse(_dequant_supported(m))

    def test_fused_probe_routes_packed_to_fallback_without_raising(self):
        wkv = [_FakeLinear(qweight=torch.zeros(8, 8, dtype=torch.int32))
               for _ in range(3)]
        self.assertFalse(_fused_commit_kv_proj_supported(wkv_linears=wkv))


if __name__ == "__main__":
    unittest.main()
