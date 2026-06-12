"""MI35x DeepSeek-V4-Pro FP4 prefill context-parallel (CP) accuracy test (8-GPU).

Shares the launch conventions of test_deepseek_v4_pro_fp4.py (same 1.6T model,
same env, same long launch timeout) but enables prefill CP via
``--enable-prefill-cp --cp-strategy interleave`` over the unified_kv backend.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro suite
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
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
# Pro is 1.6T; weight load + warmup is much longer than Flash 285B.
SERVER_LAUNCH_TIMEOUT = 5400

# Common DeepSeek-V4 env vars, aligned with test_deepseek_v4_pro_fp4.py, except
# SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton which the prefill-CP path requires.
COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_OPT_FP8_WO_A_GEMM": "false",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
    "SGLANG_OPT_USE_TOPK_V2": "false",
    "SGLANG_OPT_USE_AITER_INDEXER": "true",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
    "SGLANG_ROCM_USE_MULTI_STREAM": "false",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_EAGER_INPUT_NO_COPY": "false",
}

# FP4 variant (matches test_deepseek_v4_pro_fp4.py; V4-Pro also auto-detects it).
FP4_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "true",
}


class TestDeepseekV4ProFp4CPInterleave(CustomTestCase):
    """DeepSeek-V4-Pro FP4 unified_kv prefill CP, interleave (round-robin-split), tp=8."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_PRO_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

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
            "--disable-radix-cache",
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
            "8192",
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
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
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
                f"### test_a_gsm8k (deepseek-v4-pro-fp4-cp-interleave)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()
