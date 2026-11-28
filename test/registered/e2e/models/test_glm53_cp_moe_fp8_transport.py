"""GLM CP FP8 transport flag-off/on full-model GSM8K regression.

Prefix caching remains enabled. The four-rank transport kernel test covers
changed-input graph replay and collective contracts independently.
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-h200")


class _GlmCpTransportBase(GSM8KMixin):
    gsm8k_backend = "sgl_eval"
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 16
    gsm8k_thinking = False
    gsm8k_max_tokens = 4096
    gsm8k_temperature = 0.0
    gsm8k_score_threshold = 0.90

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("SGLANG_TEST_GLM53_MODEL_PATH") or try_cached_model(
            "zai-org/GLM-5.3-Flash"
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            env={"SGLANG_CP_MOE_FP8_TRANSPORT": str(cls.fp8_transport)},
            other_args=[
                "--tp-size",
                "4",
                "--ep-size",
                "1",
                "--attn-cp-size",
                "4",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--moe-runner-backend",
                "deep_gemm",
                "--mem-fraction-static",
                "0.75",
                "--max-total-tokens",
                "131072",
                "--max-running-requests",
                "16",
                "--chunked-prefill-size",
                "8192",
                "--cuda-graph-backend-prefill",
                "breakable",
                "--cuda-graph-max-bs-prefill",
                "8192",
                "--cuda-graph-max-bs-decode",
                "16",
                "--host",
                "127.0.0.1",
                "--random-seed",
                "42",
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)


class TestGlmCpBf16Transport(_GlmCpTransportBase, CustomTestCase):
    fp8_transport = 0


class TestGlmCpFp8Transport(_GlmCpTransportBase, CustomTestCase):
    fp8_transport = 1


if __name__ == "__main__":
    unittest.main()
