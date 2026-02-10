import unittest
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)


# RadixAttention server integration tests
class TestRadixCacheFCFS(CustomTestCase):
    """
    Testcase：Verify the scheduling policy works correctly which is set by --schedule-policy parameter.

    [Test Category] Parameter
    [Test Target] --schedule-policy
    """
    extra_args = ["--schedule-policy", "fcfs", ]

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                *cls.extra_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)


class TestRadixCacheLPM(TestRadixCacheFCFS):
    extra_args = ["--schedule-policy", "lpm", ]


class TestRadixCacheNonOverlapLPM(TestRadixCacheFCFS):
    extra_args = [
        "--schedule-policy", "lpm",
        "--disable-overlap-schedule",
    ]


if __name__ == "__main__":
    envs.SGLANG_TEST_RETRACT.set(True)
    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.set(1)
    unittest.main()
