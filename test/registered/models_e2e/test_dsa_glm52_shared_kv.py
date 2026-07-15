"""End-to-end accuracy test for DSA shared KV cache on GLM-5.2."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=720, stage="extra-b", runner_config="8-gpu-h200")


class TestGLM52DSASharedKV(DefaultServerBase, GSM8KMixin):
    model = "zai-org/GLM-5.2-FP8"
    other_args = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--attn-cp-size",
        "8",
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--enable-dsa-shared-kv-cache",
        "--dsa-prefill-backend",
        "flashmla_sparse",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--page-size",
        "64",
        "--mem-fraction-static",
        "0.85",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    gsm8k_accuracy_thres = 0.94
    gsm8k_num_questions = 500
    gsm8k_num_threads = 100
    gsm8k_num_shots = 24


if __name__ == "__main__":
    unittest.main()
