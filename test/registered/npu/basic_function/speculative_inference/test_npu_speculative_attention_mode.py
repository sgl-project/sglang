import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline, run_npu_pr_smoke
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-4-npu-a3")
register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)


class TestNpuSpeculativeAttentionMode(CustomTestCase):
    """Testcase: Verify that model inference accuracy remains uncompromised when launching the server
    with --speculative-attention-mode set to 'decode' and 'prefill' respectively.

    [Test Category] Parameter
    [Test Target] --speculative-attention-mode
    """

    def _run_gsm8k_eval(self):
        """Helper method to run GSM8K evaluation and return metrics."""
        eval_args = SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            eval_name="gsm8k",
            api="completion",
            model=QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            num_examples=1319,
            num_threads=128,
            max_tokens=512,
            num_shots=5,
            temperature=0.0,
        )
        return run_eval(eval_args)

    def test_speculative_attention_mode_decode(self):
        """Test --speculative-attention-mode decode without PD disaggregation."""
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]

        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "TRANSFORMERS_VERBOSITY": "error",
            }
        )

        process = popen_launch_server(
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )

        try:
            if _is_pr_pipeline:
                run_npu_pr_smoke(DEFAULT_URL_FOR_TEST)
            else:
                metrics = self._run_gsm8k_eval()
                self.assertGreaterEqual(
                    metrics["score"],
                    0.83,
                    f"GSM8K score {metrics['score']} below threshold 0.83",
                )
        finally:
            kill_process_tree(process.pid)

    def test_speculative_attention_mode_prefill(self):
        """Test --speculative-attention-mode prefill without PD disaggregation."""
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "prefill",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]

        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "TRANSFORMERS_VERBOSITY": "error",
            }
        )

        process = popen_launch_server(
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )

        try:
            if _is_pr_pipeline:
                run_npu_pr_smoke(DEFAULT_URL_FOR_TEST)
            else:
                metrics = self._run_gsm8k_eval()
                self.assertGreaterEqual(
                    metrics["score"],
                    0.83,
                    f"GSM8K score {metrics['score']} below threshold 0.83",
                )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
