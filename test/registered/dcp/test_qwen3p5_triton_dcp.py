import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=4800,
    suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35-triton-dcp",
    nightly=True,
)
register_cuda_ci(est_time=4800, stage="nightly", runner_config="4-gpu-b200")

QWEN35_MODEL_PATH = os.environ.get("QWEN3_5_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B-FP8")
SERVER_LAUNCH_TIMEOUT = 4800
TP_SIZE = 4
DCP_SIZE = 4
GSM8K_ACCURACY_THRESHOLD = 0.90
CHUNKED_PREFILL_SIZE = 16384
# 16384 % 5461 == 1: the 4th request's first chunk is 1 token, so its
# continuation has a prefix shorter than DCP_SIZE.
TINY_PREFIX_REQ_LEN = 5461
TINY_PREFIX_NUM_REQS = 8


class TestQwen35TritonDCPGsm8k(CustomTestCase):
    """Qwen3.5 Triton DCP (TP4/DCP4) full GSM8K accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp",
            str(TP_SIZE),
            "--dcp-size",
            str(DCP_SIZE),
            "--attention-backend",
            "triton",
            "--context-length",
            "1048576",
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
            "--disable-radix-cache",
            "--json-model-override-args",
            (
                '{"rope_scaling":{"rope_type":"yarn","factor":4.0,'
                '"original_max_position_embeddings":262144}}'
            ),
        ]
        env = os.environ.copy()
        if torch.version.hip:
            env["SGLANG_USE_AITER"] = "1"
            env["HSA_NO_SCRATCH_RECLAIM"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=32,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f'### test_a_gsm8k (qwen3.5-triton-dcp4)\n{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], GSM8K_ACCURACY_THRESHOLD)

    def test_b_prefix_shorter_than_dcp_size(self):
        """Chunked prefill leaving a 1-token first chunk must not deadlock the DCP ranks."""
        assert CHUNKED_PREFILL_SIZE % TINY_PREFIX_REQ_LEN < DCP_SIZE

        def generate(i):
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": [
                        1000 + (i + j) % 5000 for j in range(TINY_PREFIX_REQ_LEN)
                    ],
                    "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                },
                timeout=240,
            )
            resp.raise_for_status()
            return resp.json()["meta_info"]["completion_tokens"]

        with ThreadPoolExecutor(TINY_PREFIX_NUM_REQS) as pool:
            completion_tokens = list(pool.map(generate, range(TINY_PREFIX_NUM_REQS)))
        self.assertEqual(completion_tokens, [4] * TINY_PREFIX_NUM_REQS)


if __name__ == "__main__":
    unittest.main()
