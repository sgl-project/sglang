"""MI35x DeepSeek-R1-MXFP4 TP=4 GSM8K AITER MLA regression.

DeepSeek-R1 has 128 attention heads, so TP=4 gives 32 heads per rank. This
covers the AITER persistent MLA decode metadata path for the nhead=32 case.

Registry: nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-tp4 suite
"""

import os
import unittest
from types import SimpleNamespace
from typing import Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=1800,
    suite="nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-tp4",
    nightly=True,
)


SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_ACCURACY_THRESHOLD = 0.93


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run the canonical sgl-eval GSM8K benchmark."""
    metrics = run_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=base_url,
            num_examples=num_questions,
            num_threads=parallel,
            max_tokens=2048,
        )
    )
    return metrics["score"], metrics["invalid"], metrics["latency"]


class TestDeepSeekR1MXFP4TP4MI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "DEEPSEEK_R1_MXFP4_MODEL_PATH", "amd/DeepSeek-R1-MXFP4-Preview"
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_AITER_MLA_PERSIST"] = "1"

        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--attention-backend",
                "aiter",
                "--tp",
                "4",
                "--chunked-prefill-size",
                "131072",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.85",
                "--trust-remote-code",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        acc, invalid, latency = run_gsm8k_benchmark(
            self.base_url, num_questions=self.num_questions
        )
        print(f"accuracy={acc:.3f} invalid={invalid:.3f} latency={latency:.1f}s")

        if is_in_ci():
            write_github_step_summary(
                "### DeepSeek-R1-MXFP4 TP=4 GSM8K (MI35x)\n\n"
                "| Model | TP | Examples | Accuracy | Invalid | Threshold | Latency |\n"
                "| ----- | -- | -------- | -------- | ------- | --------- | ------- |\n"
                f"| {self.model} | 4 | {self.num_questions} | {acc:.3f} | "
                f"{invalid:.3f} | {GSM8K_ACCURACY_THRESHOLD:.2f} | {latency:.1f}s |\n"
            )

        self.assertGreaterEqual(acc, GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
