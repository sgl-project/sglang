import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    FR_SPEC_TOKEN_MAP_PATH,
    LLAMA_3_8B_EAGLE_WEIGHTS_PATH,
    LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
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

register_npu_ci(est_time=300, suite="nightly-8-npu-a3", nightly=True)

# Intentionally invalid path to verify EAGLE3 ignores token map and does not crash
INVALID_TOKEN_MAP_PATH = "/nonexistent/token_map.pt"


class TestNpuSpeculativeTokenMap(CustomTestCase):
    """Test --speculative-token-map with EAGLE3 (ignored) and EAGLE (enabled).

    Both cases run GSM8K evaluation to ensure accuracy does not degrade.
    """

    def test_eagle3_ignores_token_map_gsm8k(self):
        """EAGLE3 ignores token map; GSM8K accuracy should meet threshold."""
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
            "--speculative-token-map",
            INVALID_TOKEN_MAP_PATH,  # EAGLE3 should ignore this and proceed normally
            "--tp-size",
            "8",
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
            eval_args = SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                eval_name="gsm8k",
                api="completion",
                num_examples=1319,
                num_threads=128,
                max_new_tokens=512,
                num_shots=5,
                temperature=0.0,
            )
            metrics = run_eval(eval_args)
            self.assertGreaterEqual(metrics["score"], 0.86)
        finally:
            if process is not None:
                kill_process_tree(process.pid)

    def test_eagle_with_valid_token_map_gsm8k(self):
        """EAGLE (EAGLE-2) with valid token map; GSM8K accuracy should meet threshold."""

        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            LLAMA_3_8B_EAGLE_WEIGHTS_PATH,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "8",
            "--speculative-token-map",
            FR_SPEC_TOKEN_MAP_PATH,
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "float16",
        ]
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )
        process = popen_launch_server(
            LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )
        try:
            eval_args = SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                eval_name="gsm8k",
                api="completion",
                num_examples=1319,
                num_threads=128,
                max_new_tokens=512,
                num_shots=5,
                temperature=0.0,
            )
            metrics = run_eval(eval_args)
            self.assertGreaterEqual(
                metrics["score"], 0.79
            )  # adjust threshold as needed
        finally:
            if process is not None:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
