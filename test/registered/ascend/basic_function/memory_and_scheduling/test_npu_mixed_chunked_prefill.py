import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestMixedChunkedPrefill(GSM8KMixin, CustomTestCase):
    """Verify that --enable-mixed-chunk works correctly on NPU and the
    GSM8K accuracy meets the threshold.

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk;--chunked-prefill-size
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.62

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",  # DEFAULT_NPU_PAGE_SIZE = 128，we need to make sure --chunked-prefill-size%128=0
        "128",
        "--attention-backend",
        "ascend",
    ]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
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
    """Variant with --disable-radix-cache on top of mixed-chunk on NPU.

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk;--disable-radix-cache
    """

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",  # DEFAULT_NPU_PAGE_SIZE = 128，we need to make sure --chunked-prefill-size%128=0
        "128",
        "--disable-radix-cache",
        "--attention-backend",
        "ascend",
    ]


if __name__ == "__main__":
    unittest.main()
