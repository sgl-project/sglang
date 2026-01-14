import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.mamba_scheduler_strategy_test_utils import MambaSchedulerStrategyMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3NextMTP(GSM8KMixin, MambaSchedulerStrategyMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "no_buffer",
    ]

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        self._test_input_output_logprobs_match_prefill_cache_hit_helper(
            max_samples=32,
            max_new_tokens=512,
        )

    def test_input_output_logprobs_match_decode_cache_hit(self):
        self._test_input_output_logprobs_match_decode_cache_hit_helper(
            max_samples=32,
            max_new_tokens=512,
        )


class TestQwen3NextMTPTopk(GSM8KMixin, MambaSchedulerStrategyMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        self._test_input_output_logprobs_match_prefill_cache_hit_helper(
            max_samples=32,
            max_new_tokens=512,
        )

    def test_input_output_logprobs_match_decode_cache_hit(self):
        self._test_input_output_logprobs_match_decode_cache_hit_helper(
            max_samples=32,
            max_new_tokens=512,
        )

    def test_prefix_cache_branching(self):
        self._test_prefix_cache_branching_helper(64)


if __name__ == "__main__":
    unittest.main()
