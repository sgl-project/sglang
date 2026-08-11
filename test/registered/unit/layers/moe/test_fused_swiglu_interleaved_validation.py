import unittest

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _validate_fused_swiglu_interleaved,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFusedSwiGLUInterleavedValidation(CustomTestCase):
    def test_valid_configuration(self):
        _validate_fused_swiglu_interleaved(
            activation="silu",
            is_gated=True,
            has_gemm1_modifiers=False,
            has_bias=False,
            is_quantized=False,
            apply_router_weight_on_input=False,
            has_hooks=False,
            dtype=torch.bfloat16,
        )

    def test_invalid_configuration_raises(self):
        with self.assertRaisesRegex(ValueError, "incompatible fused_moe call"):
            _validate_fused_swiglu_interleaved(
                activation="gelu",
                is_gated=True,
                has_gemm1_modifiers=False,
                has_bias=False,
                is_quantized=False,
                apply_router_weight_on_input=False,
                has_hooks=False,
                dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
