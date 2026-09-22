"""DeepSeek-V4-Flash FP4 sparse prefill with attn-CP.

Launches a real DeepSeek-V4-Flash server with DSA prefill CP in
interleave mode and forces the FlashMLA sparse-prefill path through
SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1. GSM8K verifies end-to-end accuracy.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
SPARSE_PREFILL_ENV = {
    "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL": "1",
}


class TestDSV4FlashFP4SparsePrefillCP(GSM8KMixin, CustomTestCase):
    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--attn-cp-size",
                "4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
            ],
            env=SPARSE_PREFILL_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
