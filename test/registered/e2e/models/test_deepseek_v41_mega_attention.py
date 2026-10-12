import unittest

from test_deepseek_v41_flash_gb300 import (
    MODEL,
    DSV41FlashAccuracyMixin,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=900, stage="nightly", runner_config="4-gpu-gb300")


class TestDSV41MegaAttentionDSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    gsm8k_num_examples = 128
    mmlu_num_examples = 256
    gsm8k_num_threads = 16
    mmlu_num_threads = 16
    parallel_args = [
        "--attn-dp-size",
        "4",
        "--ep-size",
        "4",
        "--moe-a2a-backend",
        "none",
        "--dsv4-attn-backend",
        "flashmla_mega",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.addClassCleanup(cls.cleanup_server)
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-max-bs-decode",
                "64",
                "--max-running-requests",
                "64",
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
                *cls.parallel_args,
            ],
        )

    @classmethod
    def cleanup_server(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)


if __name__ == "__main__":
    unittest.main()
