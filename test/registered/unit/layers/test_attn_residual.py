import unittest

from sglang.srt.layers.attn_residual import _supports_attn_res_tma
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestAttnResidual(unittest.TestCase):
    def test_tma_capability_gate(self):
        self.assertFalse(_supports_attn_res_tma((9, 0)))
        self.assertTrue(_supports_attn_res_tma((10, 0)))
        self.assertTrue(_supports_attn_res_tma((10, 3)))
        self.assertTrue(_supports_attn_res_tma((11, 0)))
        self.assertFalse(_supports_attn_res_tma((12, 0)))
        self.assertTrue(_supports_attn_res_tma((13, 0)))


if __name__ == "__main__":
    unittest.main()
