"""Which renorm variant each call site gets, from the published config alone.

The fast kernels can leave TP ranks disagreeing in the last bits, and every
consumer (plain sampler, speculative verify) commits what it sampled from them
to per-rank state. The policy must therefore turn deterministic on exactly when
more than one attention-TP rank is involved, and nowhere else unless forced.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.sampling_renorm import renorm_deterministic
from sglang.srt.runtime_context import get_context, get_parallel, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestRenormPolicy(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_single_rank_is_fast(self):
        with get_context().override_server_args(tp_size=1):
            self.assertFalse(renorm_deterministic())

    def test_multi_rank_is_deterministic(self):
        with get_context().override_server_args(tp_size=2):
            self.assertEqual(get_parallel().attn_tp_size, 2)
            self.assertTrue(renorm_deterministic())

    def test_deterministic_inference_forces_on(self):
        with get_context().override_server_args(
            tp_size=1, enable_deterministic_inference=True
        ):
            self.assertTrue(renorm_deterministic())

    def test_env_overrides_both_ways(self):
        with get_context().override_server_args(tp_size=2):
            with envs.SGLANG_RENORM_DETERMINISTIC.override(False):
                self.assertFalse(renorm_deterministic())
        with get_context().override_server_args(tp_size=1):
            with envs.SGLANG_RENORM_DETERMINISTIC.override(True):
                self.assertTrue(renorm_deterministic())


if __name__ == "__main__":
    unittest.main()
