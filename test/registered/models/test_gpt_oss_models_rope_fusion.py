"""
GPT-OSS RoPE+FP8 Quant+KV Cache fusion tests.

Tests the fused FlashInfer kernel (rope_quantize_fp8_append_paged_kv_cache) in the
TRTLLM MHA backend with piecewise CUDA graph and optionally EAGLE speculative decoding.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=600,
    suite="stage-b-test-2-gpu-large",
)

GPT_OSS_MODEL = "openai/gpt-oss-120b"
GPT_OSS_EAGLE3_DRAFT_MODEL = "nvidia/gpt-oss-120b-Eagle3"

ACC_THRESHOLDS = {
    "gsm8k": 0.81,
}

BASE_ARGS = [
    "--tp",
    "2",
    "--trust-remote-code",
    "--reasoning-parser",
    "gpt-oss",
    "--kv-cache-dtype",
    "fp8_e4m3",
]


def _run_gsm8k(base_url):
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host="http://127.0.0.1",
        port=int(base_url.split(":")[-1]),
    )
    return run_eval(args)


class TestGptOssRopeFusion(CustomTestCase):
    """Test GPT-OSS accuracy with FlashInfer RoPE fusion enabled."""

    def _launch_and_eval(self, extra_args=None, extra_env=None):
        env = {"SGLANG_ENABLE_FLASHINFER_ROPE_FUSION": "1"}
        if extra_env:
            env.update(extra_env)
        for k, v in env.items():
            os.environ[k] = v

        server_args = BASE_ARGS + (extra_args or [])
        process = popen_launch_server(
            GPT_OSS_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )
        try:
            metrics = _run_gsm8k(DEFAULT_URL_FOR_TEST)
            print(f"{metrics=}")
            self.assertGreaterEqual(
                metrics["accuracy"],
                ACC_THRESHOLDS["gsm8k"],
            )
        finally:
            kill_process_tree(process.pid)
            for k in env:
                os.environ.pop(k, None)

    def test_rope_fusion_pcg(self):
        """RoPE fusion with piecewise CUDA graph (default)."""
        self._launch_and_eval()

    def test_rope_fusion_no_pcg(self):
        """RoPE fusion without piecewise CUDA graph."""
        self._launch_and_eval(extra_args=["--disable-piecewise-cuda-graph"])

    def test_rope_fusion_eagle(self):
        """RoPE fusion with EAGLE3 speculative decoding."""
        eagle_args = [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            GPT_OSS_EAGLE3_DRAFT_MODEL,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--cuda-graph-max-bs",
            "100",
            "--mem-fraction-static",
            "0.85",
        ]
        eagle_env = {
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }
        self._launch_and_eval(extra_args=eagle_args, extra_env=eagle_env)


if __name__ == "__main__":
    unittest.main()
