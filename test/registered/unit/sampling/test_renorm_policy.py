"""Which renorm variant each call site gets, from the published config alone.

The fast kernels can leave TP ranks disagreeing in the last bits; the plain
sampler has no rank-0 broadcast to hide that, speculative verify does. The
policy must therefore turn deterministic on for the sampler exactly when more
than one attention-TP rank is involved, and nowhere else unless forced.
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

    def test_single_rank_is_fast_everywhere(self):
        with get_context().override_server_args(tp_size=1):
            self.assertFalse(renorm_deterministic(ranks_agree=False))
            self.assertFalse(renorm_deterministic(ranks_agree=True))

    def test_multi_rank_sampler_is_deterministic_spec_is_not(self):
        with get_context().override_server_args(tp_size=2):
            self.assertEqual(get_parallel().attn_tp_size, 2)
            self.assertTrue(renorm_deterministic(ranks_agree=False))
            self.assertFalse(renorm_deterministic(ranks_agree=True))

    def test_deterministic_inference_forces_on(self):
        with get_context().override_server_args(
            tp_size=1, enable_deterministic_inference=True
        ):
            self.assertTrue(renorm_deterministic(ranks_agree=True))

    def test_env_overrides_both_ways(self):
        with get_context().override_server_args(tp_size=2):
            with envs.SGLANG_RENORM_DETERMINISTIC.override(False):
                self.assertFalse(renorm_deterministic(ranks_agree=False))
        with get_context().override_server_args(tp_size=1):
            with envs.SGLANG_RENORM_DETERMINISTIC.override(True):
                self.assertTrue(renorm_deterministic(ranks_agree=True))


if __name__ == "__main__":
    unittest.main()
