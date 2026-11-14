import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNonAiterBackendsKeepThePlainPath(CustomTestCase):
    """No backend but aiter may see a changed declaration.

    The read declarations key off the projection's quant method, which is true
    on any backend with a block-quant FP8 projection, CUDA included. They must
    additionally be gated on _use_aiter, or a CUDA run of Qwen3-Next would get
    a tuple where it used to get a tensor.
    """

    def test_attention_declaration_is_gated_on_use_aiter(self):
        """The gate is `_use_aiter and _linear_accepts_fp8_tuple(...)`."""
        import inspect

        import sglang.srt.models.qwen3_next as qn

        src = inspect.getsource(qn)
        gates = [
            line
            for line in src.splitlines()
            if "accepts_fp8_input = " in line and "_linear_accepts_fp8_tuple" in line
        ]
        # Both decoder layers declare a quantized attention read. Finding fewer
        # means the declaration moved and this test would check nothing.
        self.assertEqual(len(gates), 2, gates)
        for line in gates:
            self.assertIn(
                "_use_aiter and",
                line,
                "declare_attn's fp8_input must be gated on _use_aiter, "
                "otherwise CUDA installs a quantized read",
            )

    def test_select_passes_tensors_through_untouched(self):
        """A non-tuple input is returned as-is, whatever the linear is."""
        import sglang.srt.models.qwen3_next as qn

        x = torch.zeros(4, 8)
        self.assertIs(qn._select_fused_ar_input_for_linear(x, object()), x)


if __name__ == "__main__":
    unittest.main()
