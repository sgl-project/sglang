import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")

MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestRadixCacheHit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multiturn_cache_hit(self):
        run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=8,
            num_rounds=6,
            request_length=289,
            output_length=367,
        )


if __name__ == "__main__":
    unittest.main()
