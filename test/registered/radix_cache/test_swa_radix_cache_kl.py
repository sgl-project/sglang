import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "openai/gpt-oss-20b"

ACC_THRESHOLDS = {
    MODEL: {"kl_div": 0.002},
}

register_cuda_ci(est_time=100, suite="stage-b-test-large-1-gpu")


class TestSWARadixCacheKL(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "1",
                "--mem-fraction-static",
                "0.75",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        test_input_output_logprobs_match_prefill_cache_hit_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=512,
        )

    def test_input_output_logprobs_match_decode_cache_hit(self):
        test_input_output_logprobs_match_decode_cache_hit_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=2048,
        )


if __name__ == "__main__":
    unittest.main()
