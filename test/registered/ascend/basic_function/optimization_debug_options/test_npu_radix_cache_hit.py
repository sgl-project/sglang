import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestRadixCacheHit(CustomTestCase):
    """Verify Radix cache hits for multi-round sessions on the NPU."""

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multiturn_cache_hit(self):
        """Verify cache reuse with 8 clients and 6 session rounds."""

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
