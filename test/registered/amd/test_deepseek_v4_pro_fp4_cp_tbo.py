"""MI35x DeepSeek-V4-Pro FP4 CP + two-batch-overlap accuracy test.

The launch matches the unified-KV round-robin prefill-CP coverage and enables
two-batch overlap. CP-TBO keeps attention and MoE collectives on the
primary CP group, orders them through the shared TBO communication stream,
and gives each child its own CP-local metadata and output range.

The GSM8K test uses the same accuracy threshold as the CP-only configuration;
the full-server run also catches collective-order, metadata, and buffer-lifetime
failures that CPU unit tests cannot exercise.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree, set_ulimit
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=5400, suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro", nightly=True
)

DEEPSEEK_V4_PRO_FP4_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP4", "deepseek-ai/DeepSeek-V4-Pro"
)
SERVER_LAUNCH_TIMEOUT = 5400

COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_DSV4_UNIFIED_KV_FP8": "0",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "GPU_MAX_HW_QUEUES": "5",
}

FP4_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "true",
}


class TestDeepseekV4ProFp4CPInterleaveTbo(CustomTestCase):
    """DeepSeek-V4-Pro FP4 unified-KV prefill CP + TBO, tp=8."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_PRO_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        set_ulimit(65536)

        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)
        env.update(FP4_ENV_VARS)

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "1",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-two-batch-overlap",
            "--disable-radix-cache",
            "--disable-piecewise-cuda-graph",
            "--attention-backend",
            "dsv4",
            "--max-running-requests",
            "256",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.1",
            "--chunked-prefill-size",
            "32768",
            "--disable-shared-experts-fusion",
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
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=1319,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                "### test_a_gsm8k (deepseek-v4-pro-fp4-cp-interleave-tbo)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()
