"""MI30x DeepSeek-V4-Flash FP8 Accuracy Test (8-GPU)

GSM8K few-shot accuracy for DeepSeek-V4-Flash FP8 on MI30x (gfx942) ROCm 7.2.

Accuracy only: gfx942 has no DSV4 perf baseline to regress against yet, and the
MI35x suite already carries the 8k/1k throughput numbers. The GSM8K threshold
matches the MI35x FP8 test so the two architectures are directly comparable on
the same eval.

The launch config below is the gfx942 one from #36390, which is the only
DSV4-Flash FP8 configuration observed to serve on 8x MI300X. It differs from the
MI35x test in three places, each forced by gfx942:

- ``aiter`` attention and DSA backends rather than ``dsv4``.
- ``SGLANG_OPT_SWIGLU_CLAMP_FUSION=0``: the fusion dispatches a CUDA-only kernel.
- ``--moe-runner-backend triton``: the aiter fused MoE returns different logprobs
  for identical greedy requests on gfx942.

Registry: nightly-amd-accuracy-8-gpu-deepseek-v4-flash suite
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
    est_time=5400,
    suite="nightly-amd-accuracy-8-gpu-deepseek-v4-flash",
    nightly=True,
)

DEEPSEEK_V4_FP8_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_FP8_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)
SERVER_LAUNCH_TIMEOUT = 3600

ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_DSV4_FP4_EXPERTS": "false",
    "SGLANG_USE_AITER": "1",
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION": "0",
}


class TestDeepseekV4FlashFp8Mi30x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env.update(ENV_VARS)

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attention-backend",
            "aiter",
            "--dsa-prefill-backend",
            "aiter",
            "--dsa-decode-backend",
            "aiter",
            "--moe-runner-backend",
            "triton",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--max-running-requests",
            "256",
            "--mem-fraction-static",
            "0.75",
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

    def test_gsm8k(self):
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
                f"### test_gsm8k (deepseek-v4-flash-fp8, mi30x)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.91)


if __name__ == "__main__":
    unittest.main()
