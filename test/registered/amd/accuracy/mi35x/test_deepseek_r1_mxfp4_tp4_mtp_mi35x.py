"""MI35x DeepSeek-R1-MXFP4 TP=4 EAGLE GSM8K regression.

This mirrors the production-style TP=4 launch recipe with EAGLE speculative
decoding, overlap plan stream, FP8 KV cache, long context, and full GSM8K
client pressure enabled.

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
    est_time=3600,
    suite="nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-tp4",
    nightly=True,
)


SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_MTP_ACCURACY_THRESHOLD = 0.944


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
            max_tokens=16384,
        )
    )
    return metrics["score"], metrics["invalid"], metrics["latency"]


class TestDeepSeekR1MXFP4TP4MTPMI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "DEEPSEEK_R1_MXFP4_MODEL_PATH", "amd/DeepSeek-R1-MXFP4-Preview"
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.parallel = int(os.environ.get("GSM8K_PARALLEL", "1319"))

        env = os.environ.copy()
        env.update(
            {
                "SGLANG_USE_AITER": "1",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                # Retired on current main, but kept to mirror the reported launch.
                "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
                "SGLANG_AITER_FP8_PREFILL_ATTN": "1",
                "SGLANG_AITER_MLA_PERSIST": "1",
                "AITER_MXFP4_MOE_SF": "1",
                "SGLANG_INT4_WEIGHT": "0",
                "SGLANG_MOE_PADDING": "1",
                "SGLANG_SET_CPU_AFFINITY": "1",
                "SGLANG_ROCM_FUSED_DECODE_MLA": "1",
                "SGLANG_USE_ROCM700A": "1",
            }
        )

        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tensor-parallel-size",
                "4",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.9",
                "--chunked-prefill-size",
                "131072",
                "--attention-backend",
                "aiter",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--max-running-requests",
                "32",
                "--context-length",
                "200000",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        acc, invalid, latency = run_gsm8k_benchmark(
            self.base_url,
            num_questions=self.num_questions,
            parallel=self.parallel,
        )
        print(f"accuracy={acc:.3f} invalid={invalid:.3f} latency={latency:.1f}s")

        if is_in_ci():
            write_github_step_summary(
                "### DeepSeek-R1-MXFP4 TP=4 MTP GSM8K (MI35x)\n\n"
                "| Model | TP | Examples | Parallel | Accuracy | Invalid | Threshold | Latency |\n"
                "| ----- | -- | -------- | -------- | -------- | ------- | --------- | ------- |\n"
                f"| {self.model} | 4 | {self.num_questions} | {self.parallel} | "
                f"{acc:.3f} | {invalid:.3f} | {GSM8K_MTP_ACCURACY_THRESHOLD:.3f} | "
                f"{latency:.1f}s |\n"
            )

        self.assertGreaterEqual(acc, GSM8K_MTP_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
