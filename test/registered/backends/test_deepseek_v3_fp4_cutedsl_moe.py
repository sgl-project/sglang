"""Backend tests for CuteDSL MoE (FusedMoE + moe_runner, moe_a2a=none).

Exercises the CuteDSL moe_runner path with ModelOpt FP4 by launching a
server with --moe-runner-backend flashinfer_cutedsl.

Two configurations are tested:
  - EP=1, TP=4: each GPU holds all experts with TP-sharded intermediate dim
  - EP=4, TP=4: each GPU holds 1/4 of experts at full intermediate width,
    partial results combined via all-reduce (no A2A dispatch)

Requires 4 GPUs. Run from repo root with:
  python -m pytest test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py -v -s
Or via the nightly suite:
  python test/run_suite.py --hw cuda --suite nightly-4-gpu-b200 --nightly
"""

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

register_cuda_ci(est_time=900, suite="nightly-4-gpu-b200", nightly=True)

FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3-0324-FP4"
SERVER_LAUNCH_TIMEOUT = 1000
GSM8K_ACCURACY_THRESHOLD = 0.935


class TestDeepseekV3FP4CuteDSLMoE(CustomTestCase):
    """CuteDSL standard moe_runner path: flashinfer_cutedsl + modelopt_fp4, EP=1."""

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--ep",
            "1",
            "--mem-fraction-static",
            "0.75",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_cutedsl",
            "--quantization",
            "modelopt_fp4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4-cutedsl-moe)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)


class TestDeepseekV3FP4CuteDSLMoEEP4(CustomTestCase):
    """CuteDSL standard moe_runner path: flashinfer_cutedsl + modelopt_fp4, EP=TP=4."""

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--ep",
            "4",
            "--mem-fraction-static",
            "0.75",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_cutedsl",
            "--moe-a2a-backend",
            "none",
            "--quantization",
            "modelopt_fp4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4-cutedsl-moe-ep4)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
