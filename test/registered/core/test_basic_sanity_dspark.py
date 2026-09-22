import os
import tempfile
import unittest
from pathlib import Path

import requests
import torch

from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import MMLUSanityMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.spec_server_kits import SpecGrammarKit, SpecLogprobKit
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=97, stage="base-b", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"

# trtllm_mha prefill requires SM100 (Blackwell); use the Hopper-native pair elsewhere.
if is_sm100_supported():
    ATTENTION_BACKEND = "trtllm_mha"
    DRAFT_ATTENTION_BACKEND = "fa4"
else:
    ATTENTION_BACKEND = "fa3"
    DRAFT_ATTENTION_BACKEND = "fa3"


class TestBasicSanityDSpark(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    MMLUSanityMixin,
    JSONConstrainedMixin,
    SpecGrammarKit,
    SpecLogprobKit,
    CustomTestCase,
):
    served_model_name = TARGET_MODEL
    model = TARGET_MODEL

    fwd_occupancy_threshold = 60
    fwd_occupancy_max_new_tokens = 4096
    fwd_occupancy_acc_length_threshold: float = 2.0

    mmlu_score_threshold = 0.70
    mmlu_accept_length_thres = 3.0

    attention_backend = ATTENTION_BACKEND
    draft_attention_backend = DRAFT_ATTENTION_BACKEND

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                cls.attention_backend,
                "--speculative-draft-attention-backend",
                cls.draft_attention_backend,
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--enable-metrics",
                "--cuda-graph-backend-prefill=disabled",
            ],
            env={
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_DSPARK_USER_CHECKPOINT") == "1",
    "opt-in test requires the user's local target/draft checkpoints and CUDA",
)
class TestDSparkUserCheckpointCandidates(CustomTestCase):
    """User-checkpoint serving smoke, not a quality or throughput benchmark.

    Select this class explicitly to avoid running the public-checkpoint suite
    above. No remote model or trust_remote_code fallback is used here.
    """

    @classmethod
    def setUpClass(cls):
        cls.target = os.environ["SGLANG_TEST_DSPARK_TARGET_PATH"]
        cls.draft = os.environ["SGLANG_TEST_DSPARK_DRAFT_PATH"]
        for path in (cls.target, cls.draft):
            if not Path(path).is_absolute() or not Path(path).is_dir():
                raise RuntimeError(
                    f"User checkpoint must be an existing absolute local directory: {path}"
                )
        if not torch.cuda.is_available():
            raise RuntimeError("The opted-in user-checkpoint test requires NVIDIA CUDA")
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _generate(self, *, mixed=False):
        prompts = [
            "Q: What is the capital of France? Reply in one word.\nA:",
            "Q: What is 17 multiplied by 23? Reply with the number.\nA:",
            "Q: Name one primary color.\nA:",
            "Q: Complete the sequence: 2, 4, 6,\nA:",
        ]
        params = [
            {
                "temperature": 0.8 if mixed and i % 2 else 0.0,
                "top_k": -1,
                "top_p": 1.0,
                "max_new_tokens": 32,
                "ignore_eos": True,
            }
            for i in range(len(prompts))
        ]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": params,
                "return_logprob": True,
                "logprob_start_len": -1,
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), len(prompts))
        for result in results:
            self.assertIsInstance(result["text"], str)
            self.assertEqual(result["meta_info"]["completion_tokens"], 32)
            token_logprobs = result["meta_info"]["output_token_logprobs"]
            self.assertEqual(len(token_logprobs), 32)
            self.assertTrue(all(isinstance(entry[1], int) for entry in token_logprobs))
        return results

    def _run_server(self, *, candidate, graph):
        args = [
            "--tp-size",
            "1",
            "--dp-size",
            "1",
            "--max-running-requests",
            "128",
            "--disable-radix-cache",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-backend-decode",
            "full" if graph else "disabled",
            "--cuda-graph-max-bs-decode",
            "4",
            "--mem-fraction-static",
            "0.7",
        ]
        if candidate:
            args += [
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                self.draft,
                "--speculative-dspark-block-size",
                "8",
                "--speculative-dspark-markov-topk",
                "32",
                "--speculative-dspark-markov-bias-topk",
                "128",
            ]
        process = None
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
            try:
                process = popen_launch_server(
                    self.target,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                    env={
                        "SGLANG_RAGGED_VERIFY_MODE": "static",
                        "SGLANG_DSPARK_FOLDED_PROPOSAL": "1",
                    },
                    return_stdout_stderr=(log, log),
                )
                greedy = self._generate()
                if candidate:
                    self._generate(mixed=True)
                self.assertIsNone(
                    process.poll(), "server exited during candidate smoke"
                )
            finally:
                if process is not None:
                    terminate_and_kill_process_tree(process)
                log.flush()
                log.seek(0)
                logs = log.read()
            if candidate:
                self.assertRegex(
                    logs,
                    r"effective K/M=32/128[^\n]*path=candidate-triton",
                    "Candidate fallback is not a valid passing result.\n"
                    + logs[-20000:],
                )
                if graph:
                    self.assertRegex(
                        logs,
                        r"DSpark draft proposal .*folded into the draft cuda graph",
                    )
            return greedy

    def test_target_greedy_parity_and_actual_mixed_request_smoke(self):
        for graph in (False, True):
            with self.subTest(cuda_graph=graph):
                baseline = self._run_server(candidate=False, graph=graph)
                candidate = self._run_server(candidate=True, graph=graph)
                for expected, actual in zip(baseline, candidate):
                    self.assertEqual(actual["text"], expected["text"])
                    self.assertEqual(
                        [
                            entry[1]
                            for entry in actual["meta_info"]["output_token_logprobs"]
                        ],
                        [
                            entry[1]
                            for entry in expected["meta_info"]["output_token_logprobs"]
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
