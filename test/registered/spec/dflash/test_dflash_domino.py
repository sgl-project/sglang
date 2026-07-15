import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")


class TestDFlashDomino(CustomTestCase, GSM8KMixin):
    model = "Qwen/Qwen3-8B"
    draft_model = "Huang2020/Qwen3-8B-Domino-b16"
    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 4.0
    gsm8k_num_threads = 128
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "1",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-draft-attention-backend",
                "triton",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-overlap-schedule",
                "--max-running-requests",
                "64",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
