"""GSM8K coverage for FlashInfer prefill context parallelism."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


class TestFlashInferPrefillCPServer(GSM8KMixin, CustomTestCase):
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.38
    gsm8k_num_examples = 200
    gsm8k_max_tokens = 512

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-0.6B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--attn-cp-size",
                "2",
                "--attention-backend",
                "flashinfer",
                "--enable-prefill-cp",
                "--cp-strategy",
                "zigzag",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)


if __name__ == "__main__":
    unittest.main()
