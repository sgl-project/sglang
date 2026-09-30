import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")


class TestDFlashDominoFullVocab(CustomTestCase):
    model = "Qwen/Qwen3-8B"
    draft_model = "Huang2020/Qwen3-8B-Domino-b16"
    candidate_pool_size = 0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.server_log = NamedTemporaryFile(mode="w+", suffix="-domino.log")
        cls.addClassCleanup(cls.server_log.close)
        print(f"Domino server log: {cls.server_log.name}", flush=True)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            return_stdout_stderr=(cls.server_log, cls.server_log),
            other_args=[
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "2",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-domino-candidate-pool-size",
                str(cls.candidate_pool_size),
                "--speculative-draft-attention-backend",
                "triton",
                "--cuda-graph-backend-decode",
                "full",
                "--cuda-graph-max-bs-decode",
                "64",
                "--max-running-requests",
                "64",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    def test_domino_runtime(self):
        response = requests.get(self.base_url + "/server_info", timeout=10)
        response.raise_for_status()
        state = response.json()["internal_states"][0]
        self.assertEqual(state["tp_size"], 2)
        self.assertEqual(state["speculative_num_draft_tokens"], 16)
        self.assertFalse(state["disable_overlap_schedule"])
        self.assertEqual(
            state["speculative_domino_candidate_pool_size"], self.candidate_pool_size
        )
        log = Path(self.server_log.name).read_text()
        self.assertIn(
            "DFLASH Domino rollout enabled (BF16, TP=2, "
            f"block-shared candidate pool size={self.candidate_pool_size}).",
            log,
        )
        self.assertIn("Domino rollout folded into the draft cuda graph (tp=2)", log)
        self.assertIn(
            "Capture draft verify CUDA graph begin. backend=full, num_tokens_per_req=16,",
            log,
        )
        self.assertIn(
            "Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=16,",
            log,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDFlashDomino(TestDFlashDominoFullVocab, GSM8KMixin):
    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 4.0
    gsm8k_num_threads = 128
    gsm8k_num_shots = 5
    candidate_pool_size = 2048


if __name__ == "__main__":
    unittest.main()
