"""Structural checks for the sm_121 skinny GEMM plan table (no GPU needed)."""

import unittest

from sglang.kernels.ops.gemm.sm121_skinny_gemm import SM121_GEMM_PLANS
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSm121SkinnyPlans(unittest.TestCase):
    def test_plan_invariants(self):
        for (n, k), plans in SM121_GEMM_PLANS.items():
            for m, cfg in plans.items():
                with self.subTest(n=n, k=k, m=m):
                    self.assertEqual(cfg.num_rows, m)
                    self.assertTrue(1 <= m <= 16)
                    self.assertEqual(cfg.block_size % 32, 0)
                    self.assertEqual(n % cfg.outputs_per_block, 0)
                    self.assertEqual(k % (cfg.block_size * cfg.vector_width), 0)
                    if cfg.static_k is not None:
                        self.assertEqual(cfg.static_k, k)


if __name__ == "__main__":
    unittest.main()
