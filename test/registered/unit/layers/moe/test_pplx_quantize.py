"""CPU regression: the pplx FP8 dispatch path must reach the per-token-group quantizer.

FP8 is the default dispatch dtype, so ``_quantize`` runs on every pplx dispatch;
an import of a moved kernel module there fails the first MoE forward pass.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher import pplx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

QUANT = "sglang.kernels.ops.quantization.fp8_kernel.sglang_per_token_group_quant_fp8"


def _dispatcher(use_fp8):
    impl = object.__new__(pplx._PplxDispatcherImpl)
    impl.use_fp8 = use_fp8
    return impl


class TestPplxQuantize(CustomTestCase):
    def test_fp8_dispatch_quantizes_per_token_group(self):
        hidden_states = torch.randn(3, 256, dtype=torch.bfloat16)
        x_q = torch.zeros(3, 256, dtype=torch.float8_e4m3fn)
        x_s = torch.ones(3, 2, dtype=torch.float16)

        with patch(QUANT, return_value=(x_q, x_s)) as quant:
            dp_x, dp_x_scale = _dispatcher(use_fp8=True)._quantize(hidden_states)

        quant.assert_called_once_with(hidden_states, group_size=128)
        self.assertIs(dp_x, x_q)
        self.assertEqual(dp_x_scale.dtype, torch.float32)
        torch.testing.assert_close(dp_x_scale, x_s.float())

    def test_bf16_dispatch_sends_hidden_states_unquantized(self):
        hidden_states = torch.randn(3, 256, dtype=torch.bfloat16)

        with patch(QUANT) as quant:
            dp_x, dp_x_scale = _dispatcher(use_fp8=False)._quantize(hidden_states)

        quant.assert_not_called()
        self.assertIs(dp_x, hidden_states)
        self.assertIsNone(dp_x_scale)


if __name__ == "__main__":
    unittest.main()
