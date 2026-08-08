"""MI35x DeepSeek-V4-Pro FP4 prefill context-parallel (CP) + two-batch-overlap (TBO)
accuracy test (8-GPU).

Same launch conventions as test_deepseek_v4_pro_fp4_cp.py (prefill CP over the
unified_kv backend via ``--enable-prefill-cp --cp-strategy interleave``), plus
``--enable-two-batch-overlap``. This exercises the CP TBO op strategy
(``op_cp_gather`` / ``op_cp_moe`` / ``op_cp_combine`` driven by
``DeepseekV4Model._forward_layers_tbo_cp``), which splits each prefill batch into
two token-range ubatches, round-robin splits each one across the CP group
independently, and overlaps one ubatch's CP MoE all-gather / reduce-scatter with
the other ubatch's attention + expert compute.

The overlap runs on a duplicate CP communicator (``attn_cp_overlap``) so it can
execute concurrently with the attention-internal CP all-gathers, which stay on
the compute stream and the primary CP communicator. This test guards that the
combination stays numerically equivalent to CP-only (>0.92 on GSM8K, same bar as
test_deepseek_v4_pro_fp4_cp.py) and that neither the per-ubatch CP metadata setup
nor the concurrent CP communicators deadlock or corrupt state.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro suite
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
# Pro is 1.6T; weight load + warmup is much longer than Flash 285B.
SERVER_LAUNCH_TIMEOUT = 5400

# Matches test_deepseek_v4_pro_fp4_cp.py; prefill CP requires unified_kv_triton.
COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    # ROCm HSA-resource stability for TBO at high concurrency.
    "GPU_MAX_HW_QUEUES": "5",
}

# FP4 variant (matches test_deepseek_v4_pro_fp4.py; V4-Pro also auto-detects it).
FP4_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "true",
}


class TestDeepseekV4ProFp4CPInterleaveTbo(CustomTestCase):
    """DeepSeek-V4-Pro FP4 unified_kv prefill CP (round-robin-split) + TBO, tp=8."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_PRO_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        # GSM8K below drives 1319 concurrent requests, and the launched server
        # inherits the fd limit of this process, so raise it before popen.
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
            # TBO halves the per-ubatch MoE rows, so it only pays off once the
            # chunk is large; 32768 also keeps num_q_tokens under the compress
            # prefill plan's uint16 token limit for a single full chunk.
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
            num_threads=1319,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_a_gsm8k (deepseek-v4-pro-fp4-cp-interleave-tbo)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            # CP + TBO must stay numerically equivalent to CP-only (same bar as
            # test_deepseek_v4_pro_fp4_cp.py).
            self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()
