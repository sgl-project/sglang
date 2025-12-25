import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.kl_test_utils import KLTestMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3Next(GSM8KMixin, KLTestMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
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
