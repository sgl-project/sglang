"""DeepSeek-V4-Flash MXFP4 KV cache end-to-end test (8-GPU CUDA).

Launches a real server with ``--kv-cache-dtype fp4_e2m1
--fp4-kv-cache-recipe mxfp4`` — exercising the full pool configurator,
DSV4 attention backend, compressor storage, and CUDA-graph capture
stack — and runs GSM8K accuracy. This covers the integration surface
kernel-level unit tests cannot reach (scheduler metadata lifecycle,
pool sizing, HiSparse incompatibility guards).

Registry: extra-b 8-GPU CUDA (nightly-class runtime).
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-h200")

MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_MXFP4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"
)
SERVER_LAUNCH_TIMEOUT = 3600


class TestDeepseekV4Mxfp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attention-backend",
            "dsv4",
            "--kv-cache-dtype",
            "fp4_e2m1",
            "--fp4-kv-cache-recipe",
            "mxfp4",
            "--moe-runner-backend",
            "marlin",
            "--max-running-requests",
            "8",
            "--mem-fraction-static",
            "0.85",
            "--page-size",
            "256",
            "--tool-call-parser",
            "deepseekv4",
            "--reasoning-parser",
            "deepseek-v4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=os.environ.copy(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        # `a` prefix to run first (alphabetical) and warm up the server.
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            parallel=200,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)

        if is_in_ci():
            write_github_step_summary(
                f"### gsm8k (deepseek-v4-flash-mxfp4)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
        self.assertGreater(metrics["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
