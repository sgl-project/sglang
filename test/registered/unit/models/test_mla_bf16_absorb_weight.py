"""Verify MLA absorb preserves BF16 weights with an identity host scale."""

import unittest

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    _absorb_weight_bf16,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaBf16AbsorbWeight(CustomTestCase):
    def test_unit_host_scale_skips_multiply(self):
        weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        assert _absorb_weight_bf16(weight, 1.0) is weight
        assert _absorb_weight_bf16(weight, 1) is weight
        assert (
            _absorb_weight_bf16(weight, torch.ones((), dtype=torch.float32))
            is not weight
        )
        scaled = _absorb_weight_bf16(weight, 2.0)
        assert torch.equal(scaled, weight * 2)


if __name__ == "__main__":
    unittest.main()
