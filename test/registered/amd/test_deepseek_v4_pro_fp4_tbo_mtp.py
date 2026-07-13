"""MI35x DeepSeek-V4-Pro FP4 + non-EP DP two-batch-overlap (TBO) + MTP test (8-GPU)

End-to-end accuracy test for DeepSeek-V4-Pro (1.6T) FP4 with the non-EP DP
two-batch-overlap path on MI35x ROCm 7.2.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-mtp suite
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=14400, suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro-mtp", nightly=True
)

DEEPSEEK_V4_PRO_FP4_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP4", "deepseek-ai/DeepSeek-V4-Pro"
)
# Pro is 1.6T; weight load + warmup is much longer than Flash 285B.
SERVER_LAUNCH_TIMEOUT = 5400
FLASHMLA_BACKEND = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")


COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_HACK_FLASHMLA_BACKEND": FLASHMLA_BACKEND,
    "AITER_BF16_FP8_MOE_BOUND": "0",
    # DP TP-MoE collective path that non-EP DP TBO overlaps.
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_DP_USE_REDUCE_SCATTER": "1",
    "SGLANG_SHARED_EXPERT_TP1": "1",
    "SGLANG_DP_SHARED_EXPERT_LOCAL": "1",
    # ROCm HSA-resource stability for TBO at high concurrency.
    "GPU_MAX_HW_QUEUES": "5",
}

# FP4 variant
FP4_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "true",
}


class TestDeepseekV4ProFp4TboMTP(CustomTestCase):
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
            # DP attention + TBO: non-EP DP TP-MoE two-batch-overlap. DP TBO is
            # selected because moe_a2a_backend stays 'none'; no opt-in env needed.
            "--dp",
            "8",
            "--enable-dp-attention",
            "--enable-prefill-delayer",
            "--enable-two-batch-overlap",
            "--disable-radix-cache",
            "--attention-backend",
            "dsv4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            # MTP / EAGLE speculative decoding (NextN head from the base model).
            # With spec on, target-verify batches also report is_extend(); the
            # prefill TBO gate is is_extend_without_speculative() (PR #30238).
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--max-running-requests",
            "512",
            "--cuda-graph-max-bs",
            "512",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.15",
            # global chunk; DP-attention divides by dp_size=8 -> 8192/rank.
            "--chunked-prefill-size",
            "65536",
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

    def test_gsm8k_tbo_mtp(self):
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
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v4-pro-fp4 DP+TBO+MTP, {FLASHMLA_BACKEND})\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
            # TBO+MTP must not regress accuracy vs the non-TBO MTP baseline.
            self.assertGreater(metrics["accuracy"], 0.91)


if __name__ == "__main__":
    unittest.main()
