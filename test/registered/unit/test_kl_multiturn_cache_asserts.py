"""Cache assertions must account for the preceding request's retractions."""

import unittest
from unittest.mock import patch

from sglang.test import kl_multiturn_utils as kl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def response(cached_tokens, num_retractions=0):
    return {
        "output_ids": [1] * 512,
        "meta_info": {
            "cached_tokens": cached_tokens,
            "num_retractions": num_retractions,
            "output_token_logprobs": [[-0.1, 1, None]] * 512,
        },
    }


class TestMambaDecodeCacheAssert(unittest.TestCase):
    def setUp(self):
        self.check = kl.make_mamba_decode_assert(track_interval=128)

    def test_retracted_prefill_checkpoint(self):
        # CI: 4391 + floor(491 / 64) * 64 = 4839.
        # Reproduction: 4372 + floor(509 / 64) * 64 = 4820.
        for cached_tokens in (4839, 4820):
            with self.subTest(cached_tokens=cached_tokens):
                self.check(
                    response(cached_tokens),
                    4391,
                    512,
                    "retracted",
                    previous_num_retractions=1,
                )
                with self.assertRaises(AssertionError):
                    self.check(response(cached_tokens), 4391, 512, "unretracted")

    def test_retraction_allowance_is_less_than_one_prefill_chunk(self):
        self.check(response(4801), 4391, 512, "boundary", previous_num_retractions=2)
        with self.assertRaises(AssertionError):
            self.check(response(4800), 4391, 512, "too_old", previous_num_retractions=2)

    def test_current_request_retraction_does_not_relax_previous_checkpoint(self):
        with self.assertRaises(AssertionError):
            self.check(response(4820, num_retractions=1), 4391, 512, "current_only")

    def test_no_output_and_full_attention_remain_strict(self):
        with self.assertRaises(AssertionError):
            self.check(response(99), 100, 0, "no_output", previous_num_retractions=1)
        with self.assertRaises(AssertionError):
            kl.default_decode_cache_assert(
                response(4902),
                4391,
                512,
                "full_attention",
                previous_num_retractions=1,
            )

    def test_multiturn_helpers_use_previous_retractions_per_sample(self):
        helpers = (
            kl.test_input_output_logprobs_match_helper,
            kl.test_input_output_logprobs_match_prefill_cache_hit_helper,
            kl.test_input_output_logprobs_match_decode_cache_hit_helper,
        )
        for helper in helpers:
            with self.subTest(helper=helper.__name__):
                # The retracted sample swaps between turns. Looking at the
                # current response or a batch-wide count would be incorrect.
                turns = [
                    [response(4391, 1), response(4391)],
                    [response(4820), response(4864, 1)],
                    [response(5376), response(5332)],
                ]
                if (
                    helper
                    is kl.test_input_output_logprobs_match_prefill_cache_hit_helper
                ):
                    turns.insert(0, [])  # Cache-seeding response is unused.
                with (
                    patch.object(kl, "_flush_cache"),
                    patch.object(kl, "_generate", side_effect=turns),
                    patch.object(kl, "_replay_and_compare_kl") as replay,
                ):
                    helper(
                        "http://unused",
                        "model",
                        0.005,
                        [[1] * 4391, [2] * 4391],
                        max_new_tokens=512,
                        turn_suffixes=[[[3], [4]], [[5], [6]]],
                        assert_decode_cached_tokens=self.check,
                    )
                replay.assert_called_once()
                self.assertEqual(len(replay.call_args.args[3]), 2)
                self.assertEqual(len(replay.call_args.args[4][0]), 512)


if __name__ == "__main__":
    unittest.main()
