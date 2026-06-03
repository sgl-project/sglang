"""
NVIDIA Nemotron 3 Nano 30B-A3B (BF16, hybrid MoE + Mamba): simple text Q&A on
XPU (OpenAI /v1), same shape as ``test_gemma_4_e2b.py``.

Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

  - XPU test runs when Intel XPU is available.
  - Requires tp=4 (30B params with ~3B active; ~60 GB weights sharded 4-way
    → ~15 GB per rank). On 24 GB Arc Pro B60 that leaves headroom for the
    Mamba state cache + KV pool.

Run from test/registered::

  python3 -m unittest xpu.test_nvidia_nemotron_3_nano.TestNemotron3Nano30BXPU.test_simple_qa

A single end-to-end test (``test_simple_qa``) verifies the model boots
on XPU and returns a coherent reply. On failure the assertion message
includes the model's actual output.

Server is started with ``sglang serve`` (``--model-impl sglang``).
"""

from __future__ import annotations

import os
import unittest

import openai
import torch

from sglang.srt.utils.common import is_xpu
from sglang.test.test_utils import CustomTestCase
from sglang.test.vlm_utils import (
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

LAUNCH_TIMEOUT = 1200

# 30B-A3B hybrid (Mamba + attention + MoE) model: requires tp=4 across XPU
# devices. The BF16 variant's weights are ~60 GB, sharded 4-way → ~15 GB per
# rank of weights. On a 24 GB Arc Pro B60 partition this leaves ~9 GB headroom
# per rank for the Mamba state cache, KV pool, and cuda-graph buffers.
#
# ``qwen3_coder`` tool-call parser and ``deepseek-r1`` reasoning parser match
# the registered 2-GPU BF16 configuration for this model.
XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--tp=4",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--page-size",
    "64",
    "--attention-backend",
    "intel_xpu",
    "--model-impl",
    "sglang",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "deepseek-r1",
    "--mem-fraction-static",
    "0.85",
    "--context-length",
    "8192",
    "--chunked-prefill-size",
    "1024",
    "--max-running-requests",
    "8",
    "--cuda-graph-max-bs",
    "8",
]

# Standard sglang e2e Q&A prompt (see test_openai_server.py::run_chat_completion).
_SIMPLE_QA_PROMPT = "What is the capital of France? Answer in a few words."


def _simple_text_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SIMPLE_QA_PROMPT},
            ],
        }
    ]


def _empty_xpu_cache() -> None:
    """Release cached XPU allocations so back-to-back tests start clean."""
    if torch.xpu.is_available():
        torch.xpu.empty_cache()


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestNemotron3Nano30BXPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        os.environ["SGLANG_USE_SGL_XPU"] = "1"

        _empty_xpu_cache()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            api_key=cls.api_key,
            other_args=list(XPU_SERVER_ARGS),
            # Pass a non-"auto" value so popen_launch_server skips device
            # auto-detection; the server always runs on XPU via the hardcoded
            # --device xpu in XPU_SERVER_ARGS.
            device="xpu",
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        _empty_xpu_cache()

    def test_simple_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=_simple_text_messages(),
            temperature=0,
            max_tokens=256,
        )
        msg = response.choices[0].message
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        combined = f"{text} {reasoning}".strip()

        self.assertEqual(msg.role, "assistant", f"unexpected role; got: {combined!r}")
        self.assertGreater(len(combined), 0, "empty reply from model")
        self.assertIn(
            "paris", combined.lower(), f"expected `Paris` in reply, got: {combined!r}"
        )
        self.assertIsNotNone(response.usage)
        self.assertGreater(
            response.usage.completion_tokens,
            0,
            f"no tokens generated; got: {combined!r}",
        )


from sglang.test.ci.ci_register import register_xpu_ci

# Requires tp=4; XPU CI only has a 1-GPU runner, so register but keep disabled.
register_xpu_ci(
    est_time=600,
    suite="stage-b-test-1-gpu-xpu",
    disabled="requires tp=4; no 4-GPU XPU CI runner",
)

if __name__ == "__main__":
    unittest.main()
