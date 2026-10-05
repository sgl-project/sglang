"""
E2E test: verify DSA topk_prefill_bf16 kernel does not crash with
illegal memory access when kTopK > old kMaxTies (bug 85883409).

Run:
    # Requires 8× ZW-M890P GPUs with a GLM-5.1 FP8-Channelwise model
    export MODEL_PATH=<path-to-GLM-5.1-FP8-Channelwise>
    export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
    pytest test/ppu/e2e/test_dsa_topk_bf16_prefill_crash.py -v

Prerequisites:
    - sglang server running with DSA backend, tp8, attn-cp-size=8
    - Or the test will start server internally (slow, ~10min with DG JIT)

Affected models/dtypes/platforms:
    - GLM-5.1 FP8-Channelwise / GLM-5.2 MXFP4-FP8 (index_topk=2048)
    - ZW-M890P (PPU 890P, 8×144GB)
    - DSA attention backend with flashmla_sparse prefill
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# Model path must be provided via MODEL_PATH env var.
# Any GLM-5.x FP8-Channelwise or MXFP4-FP8 checkpoint with index_topk=2048 works.
MODEL_PATH = os.environ.get("MODEL_PATH")
if MODEL_PATH is None:
    raise EnvironmentError(
        "MODEL_PATH env var is required. Set it to a GLM-5.1 FP8-Channelwise "
        "(or GLM-5.2 MXFP4-FP8) checkpoint directory."
    )
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

# Minimum prefill length to trigger the topk_prefill_bf16 kernel path.
# The bug manifests when index_topk=2048 > old kMaxTies=1024 and the
# input is long enough to exercise the BF16 topk prefill codepath.
MIN_PREFILL_TOKENS = 20000


def _build_long_prompt(num_tokens: int) -> str:
    """Build a prompt that is approximately *num_tokens* tokens long.

    Uses a repeating pattern of short words (≈1 token each) so the
    actual token count is close to len(words).
    """
    word = "hello "
    # Each "hello " is ~1-2 tokens; over-generate slightly to ensure we
    # exceed the target after tokenisation.
    return word * num_tokens


class TestDsaTopkBf16PrefillCrash(CustomTestCase):
    """E2E: verify topk_prefill_bf16 kernel with kMaxTies=kTopK does not
    crash on long prefill (bug 85883409).

    The test starts a sglang server with DSA backend and sends a long
    prefill request.  If the old kMaxTies=1024 bug is present the CUDA
    kernel will trigger an illegal memory access and the server process
    will crash.
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--attention-backend",
            "dsa",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
            "--trust-remote-code",
            "--max-running-requests",
            "4",
            "--chunked-prefill-size",
            "65536",
            "--num-continuous-decode-steps",
            "1",
            "--dist-timeout",
            "3600",
            "--attn-cp-size",
            "8",
            "--enable-dsa-prefill-context-parallel",
            "--dsa-prefill-cp-mode",
            "round-robin-split",
            "--watchdog-timeout",
            "3600",
            "--dsa-prefill-backend",
            "flashmla_sparse",
            "--dsa-decode-backend",
            "flashmla_kv",
            "--disable-custom-all-reduce",
            "--enforce-disable-flashinfer-allreduce-fusion",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs",
            "4",
            "--disable-piecewise-cuda-graph",
        ]

        env = os.environ.copy()
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

    def test_long_prefill_no_illegal_memory_access(self):
        """Send a long prefill request and verify 200 OK with no CUDA error."""
        prompt = _build_long_prompt(MIN_PREFILL_TOKENS)
        response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 1,
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
        # The server must return generated text (even if just 1 token).
        self.assertIn("text", body, f"Response missing 'text' key: {body}")

    def test_server_alive_after_long_prefill(self):
        """After the long prefill, the server should still be healthy."""
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
