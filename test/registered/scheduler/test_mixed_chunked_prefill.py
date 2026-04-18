import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=176, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")


class TestMixedChunkedPrefill(GSM8KMixin, CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.62

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "32",
    ]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.extra_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestMixedChunkedPrefillNoRadixCache(TestMixedChunkedPrefill):
    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "32",
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
