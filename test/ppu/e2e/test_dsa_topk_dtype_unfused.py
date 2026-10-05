"""
E2E test: verify DSA fast_topk_v2 (unfused path) does not crash with
"expected Float but found BFloat16" when SGLANG_DSA_FUSE_TOPK=0
(bug 85825908).

Run:
    # Requires 8× ZW-M890P GPUs with a GLM-5.2 MXFP4-FP8 model
    export MODEL_PATH=<path-to-GLM-5.2-MXFP4-FP8>
    export SGLANG_DSA_FUSE_TOPK=0
    export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
    pytest test/ppu/e2e/test_dsa_topk_dtype_unfused.py -v

Prerequisites:
    - 8× ZW-M890P (PPU 890P, 8×144GB) with NSA + EAGLE enabled
    - Or the test will start server internally (slow, ~10min with DG JIT)

Affected models/dtypes/platforms:
    - GLM-5.2 MXFP4-FP8 (FP4 indexer path, force_unfused_topk)
    - ZW-M890P (PPU 890P, 8×144GB)
    - NSA attention backend with EAGLE speculative decoding
    - Triggered when SGLANG_DSA_FUSE_TOPK=0 forces unfused topk path
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# Model path must be provided via MODEL_PATH env var.
# A GLM-5.2 MXFP4-FP8 checkpoint is required to exercise the FP4 indexer path.
MODEL_PATH = os.environ.get("MODEL_PATH")
if MODEL_PATH is None:
    raise EnvironmentError(
        "MODEL_PATH env var is required. Set it to a GLM-5.2 MXFP4-FP8 "
        "checkpoint directory."
    )
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000


def _build_short_prompt() -> str:
    """Build a short prompt for basic inference verification.

    We only need to verify the dtype path is correct, so a short prompt
    suffices — the bug manifests at warmup / first forward, not dependent
    on sequence length.
    """
    return "What is the capital of France?"


class TestDsaTopkDtypeUnfused(CustomTestCase):
    """E2E: verify FP4 indexer emits fp32 logits when force_unfused_topk
    is active, so fast_topk_v2 does not crash with dtype mismatch
    (bug 85825908).

    The test starts a sglang server with NSA backend + EAGLE speculative
    decoding and SGLANG_DSA_FUSE_TOPK=0.  If the old bug is present,
    the FP4 indexer emits bf16 logits → fast_topk_v2 expects fp32 →
    RuntimeError: expected Float but found BFloat16.
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--dp-size",
            "8",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--attention-backend",
            "nsa",
            "--nsa-prefill-backend",
            "flashmla_sparse",
            "--nsa-decode-backend",
            "flashmla_kv",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "16",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--speculative-attention-mode",
            "decode",
            "--disable-piecewise-cuda-graph",
            "--disable-custom-all-reduce",
            "--disable-shared-experts-fusion",
            "--trust-remote-code",
            "--dist-timeout",
            "3600",
            "--watchdog-timeout",
            "3600",
        ]

        env = os.environ.copy()
        env["SGLANG_DSA_FUSE_TOPK"] = "0"  # Force unfused topk path
        env.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_unfused_topk_no_dtype_crash(self):
        """Send inference request and verify no dtype mismatch crash."""
        prompt = _build_short_prompt()
        response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 8,
                    "temperature": 0,
                },
            },
            timeout=600,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Server returned {response.status_code}: {response.text[:500]}",
        )
        body = response.json()
        self.assertIn("text", body, f"Response missing 'text' key: {body}")

    def test_server_alive_after_inference(self):
        """After inference, the server should still be healthy."""
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
