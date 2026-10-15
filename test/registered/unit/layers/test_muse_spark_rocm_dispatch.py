"""CPU-safe coverage for Muse Spark's ROCm expert integration."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.rocm_muse_spark.model import (
    MuseSpark1xMxfp4Experts,
    _moe_backend,
    _want_fused_moe,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMuseSparkRocmDispatch(CustomTestCase):
    def test_mxfp4_parameter_shapes(self):
        experts = MuseSpark1xMxfp4Experts(
            num_local_experts=2,
            expert_hidden_size=64,
            intermediate_size=96,
        )
        self.assertEqual(experts.w13_weight.shape, (2, 192, 32))
        self.assertEqual(experts.w13_weight_scale.shape, (2, 192, 2))
        self.assertEqual(experts.w2_weight.shape, (2, 64, 48))
        self.assertEqual(experts.w2_weight_scale.shape, (2, 64, 3))
        self.assertEqual(experts.w13_weight.dtype, torch.uint8)

    def test_backend_override(self):
        with patch.dict(os.environ, {"MUSE_MOE_BACKEND": "native"}):
            self.assertEqual(_moe_backend(), "native")
            self.assertFalse(_want_fused_moe())
        with patch.dict(os.environ, {"MUSE_MOE_BACKEND": "triton"}):
            self.assertEqual(_moe_backend(), "triton")
            self.assertTrue(_want_fused_moe())
        with patch.dict(os.environ, {"MUSE_MOE_BACKEND": "flydsl"}):
            self.assertEqual(_moe_backend(), "flydsl")
            self.assertTrue(_want_fused_moe())

    def test_invalid_backend_is_rejected(self):
        with (
            patch.dict(os.environ, {"MUSE_MOE_BACKEND": "invalid"}),
            self.assertRaisesRegex(ValueError, "must be one of"),
        ):
            _moe_backend()


if __name__ == "__main__":
    unittest.main()
