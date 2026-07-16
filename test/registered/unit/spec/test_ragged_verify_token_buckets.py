import unittest

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import build_ragged_verify_token_buckets
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRaggedVerifyTokenBuckets(unittest.TestCase):
    def test_default_to_full_blocks(self):
        with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.override(0):
            self.assertEqual(
                build_ragged_verify_token_buckets(
                    capture_bs=[1, 2, 4], num_tokens_per_req=8
                ),
                [8, 16, 32],
            )

    def test_env_can_add_fine_grained_tiers(self):
        with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.override(8):
            with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MIN_TOKENS.override(1):
                self.assertEqual(
                    build_ragged_verify_token_buckets(
                        capture_bs=[1, 2, 4], num_tokens_per_req=8
                    ),
                    [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
                )

    def test_env_can_skip_tiny_tiers(self):
        with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.override(8):
            with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MIN_TOKENS.override(2):
                self.assertEqual(
                    build_ragged_verify_token_buckets(
                        capture_bs=[1, 2, 4], num_tokens_per_req=8
                    ),
                    [2, 3, 4, 5, 6, 7, 8, 16, 32],
                )

    def test_explicit_tiers_override_env(self):
        with envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.override(0):
            self.assertEqual(
                build_ragged_verify_token_buckets(
                    capture_bs=[1, 2, 4],
                    num_tokens_per_req=8,
                    fine_grained_min_tokens=3,
                    fine_grained_max_tokens=6,
                ),
                [3, 4, 5, 6, 8, 16, 32],
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
