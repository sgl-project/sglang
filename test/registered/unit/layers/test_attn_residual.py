import unittest
from unittest.mock import patch

import sglang.srt.layers.attn_residual as attn_residual
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

    def test_non_cuda_skips_capability_query(self):
        with (
            patch.object(attn_residual, "is_cuda", return_value=False),
            patch.object(attn_residual.torch.cuda, "get_device_capability") as query,
        ):
            self.assertFalse(attn_residual._use_fast(7168))
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
