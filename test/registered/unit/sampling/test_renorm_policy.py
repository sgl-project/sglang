"""Which renorm variant runs, from the published config alone.

The fast kernels can leave TP ranks disagreeing in the last bits, and every
consumer (plain sampler, speculative verify) commits what it sampled from them
to per-rank state. The policy therefore turns deterministic on whenever more
than one rank samples the same batch (attention TP or context parallelism), and
under --enable-deterministic-inference; a single rank, or attention DP with one
attention TP and CP rank per DP group, keeps the fast kernels.
"""

import unittest

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

    def test_context_parallel_is_deterministic(self):
        # Context parallelism leaves attn_tp_size at 1 while its CP peers sample
        # the same batch, so they must agree.
        with get_context().override_server_args(tp_size=2, attn_cp_size=2):
            self.assertEqual(get_parallel().attn_tp_size, 1)
            self.assertTrue(renorm_deterministic())

    def test_dp_attention_with_context_parallel_is_deterministic(self):
        # CP peers of a DP rank sit outside its attention-TP sync group.
        with get_context().override_server_args(
            tp_size=4, attn_dp_size=2, attn_cp_size=2
        ):
            self.assertEqual(get_parallel().attn_tp_size, 1)
            self.assertTrue(renorm_deterministic())

    def test_dp_attention_with_one_rank_per_group_is_fast(self):
        # Every DP rank samples only its own requests: nothing to agree with.
        with get_context().override_server_args(tp_size=2, attn_dp_size=2):
            self.assertTrue(get_parallel().attn_dp_enabled)
            self.assertEqual(get_parallel().attn_tp_size, 1)
            self.assertFalse(renorm_deterministic())

    def test_dp_attention_with_attention_tp_is_deterministic(self):
        with get_context().override_server_args(tp_size=4, attn_dp_size=2):
            self.assertEqual(get_parallel().attn_tp_size, 2)
            self.assertTrue(renorm_deterministic())

    def test_deterministic_inference_forces_on(self):
        with get_context().override_server_args(
            tp_size=1, enable_deterministic_inference=True
        ):
            self.assertTrue(renorm_deterministic())


if __name__ == "__main__":
    unittest.main()
