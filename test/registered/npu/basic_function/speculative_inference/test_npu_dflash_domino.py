import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from sglang.test.ascend.npu_eval_accuracy_kit import NPUGSM8KMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="base-b-test-2-npu-a3")

# Domino draft model. Community HF checkpoint; SGLang resolves the id lazily on
# the test host (matching the CUDA e2e test), so keep it as the raw repo id
# rather than a model-scope cache path.
QWEN3_8B_DOMINO_DRAFT_WEIGHTS_PATH = "Huang2020/Qwen3-8B-Domino-b16"


class TestNpuDFlashDominoFullVocab(CustomTestCase):
    """Runtime smoke test: DFLASH Domino server startup + draft config state.

    Verifies the NPU server launches with the Domino draft projector modules and
    the ``--speculative-domino-candidate-pool-size`` value is wired through to
    the worker, without running a full accuracy eval.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    draft_model = QWEN3_8B_DOMINO_DRAFT_WEIGHTS_PATH
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
                "ascend",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-domino-candidate-pool-size",
                str(cls.candidate_pool_size),
                "--disable-cuda-graph",
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
        self.assertEqual(
            state["speculative_domino_candidate_pool_size"], self.candidate_pool_size
        )
        log = Path(self.server_log.name).read_text()
        self.assertIn(
            "DFLASH Domino rollout enabled (BF16, TP=2, "
            f"block-shared candidate pool size={self.candidate_pool_size}).",
            log,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestNpuDFlashDomino(TestNpuDFlashDominoFullVocab, NPUGSM8KMixin):
    """GSM8K accuracy + speculative accept-length check for DFLASH Domino.

    Uses the block-shared candidate pool (size 2048) for draft rollouts. The
    thresholds match the CUDA baseline: NPU Domino reproduces the target's GSM8K
    accuracy while keeping the average speculation accept-length above 4.
    """

    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 4.0
    gsm8k_num_threads = 128
    gsm8k_num_shots = 5
    candidate_pool_size = 2048


if __name__ == "__main__":
    unittest.main()
