import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=443, stage="extra-b", runner_config="4-gpu-b200")

GLM52_NVFP4_MODEL_PATH = "nvidia/GLM-5.2-NVFP4"


class TestGLM52CPInterleave(GSM8KMixin, CustomTestCase):
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_examples = 500
    gsm8k_num_threads = 32
    gsm8k_num_shots = 20
    gsm8k_accept_length_thres = 3

    @classmethod
    def setUpClass(cls):
        cls.model = GLM52_NVFP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--attn-cp-size",
                "4",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-frac",
                "0.85",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
