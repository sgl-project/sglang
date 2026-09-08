"""MI30x DeepSeek-V4-Flash FP8 Accuracy Test (8-GPU)

GSM8K few-shot accuracy for DeepSeek-V4-Flash FP8 on MI30x (gfx942) ROCm 7.2.

The launch config is the cookbook's MI300X Flash FP8 low-latency single-node
recipe, verbatim -- the one the docs tell users to run, marked `verified: true`
in docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx. Of the three published
MI300X strategies it is the only TP-only one (balanced and high-throughput add
DP attention and the prefill delayer), so it is the narrowest cell that still
covers the gfx942 serving path. Keep this in sync with that cell.

The cell as published could not load this checkpoint; the fix is in the
cookbook PR and is the SGLANG_DSV4_FP4_EXPERTS entry below.

Accuracy only: gfx942 has no DSV4 perf baseline to regress against yet, and the
MI35x suite already carries the 8k/1k throughput numbers.

Unlike the MI35x FP8 test this recipe also runs `--kv-cache-dtype fp8_e4m3` and
EAGLE, so it is the gfx942 MLA + KV-FP8 signal as well. That extra quantization
is also why the threshold below is 0.90 rather than the 0.91 the MI35x FP8 test
uses: measured 0.9174 and 0.9257 here, a spread wide enough that 0.91 would sit
only ~2 sd off the mean and fail a good build roughly one night in forty. 0.90
keeps ~2 points of headroom and still catches every regression worth gating on
-- the failure modes this guards against (see #36390) collapse output entirely
rather than shaving a point.

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
    # The routed experts of this checkpoint are FP8, not mxfp4-packed. The env
    # default is mxfp4 and the auto-detect fallback reads the safetensors header
    # from the local HF cache, so it returns None on a runner that has not pulled
    # the weights yet and the wrong default wins -- weight load then dies on a
    # factor-of-2 shape mismatch. Set it rather than depend on cache state.
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "AITER_BF16_FP8_MOE_BOUND": "0",
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
            "dsv4",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.1",
            "--disable-shared-experts-fusion",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--chunked-prefill-size",
            "8192",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
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
            self.assertGreater(metrics["accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
