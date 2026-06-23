"""
Gemma 4 E2B-it: simple text Q&A on XPU (OpenAI /v1), same shape as
``test_deepseek_coder_v2_lite_instruct.py``.

Model card: https://huggingface.co/google/gemma-4-E2B-it

  - XPU test runs when Intel XPU is available.

Run from test/srt::

  python3 -m unittest xpu.test_gemma_4_e2b.TestGemma4E2BXPU.test_simple_qa

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

MODEL = "google/gemma-4-E2B-it"

LAUNCH_TIMEOUT = 900

# The -it model ships its own chat_template.jinja, so no --chat-template needed.
# E2B model: single-rank for small model on XPU.
XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--tp=1",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.70",
    "--attention-backend",
    "intel_xpu",
    "--model-impl",
    "sglang",
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
class TestGemma4E2BXPU(CustomTestCase):
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
            max_tokens=96,
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

# Single e2e test: boot + a short Q&A.
register_xpu_ci(
    est_time=240,
    suite="stage-b-test-1-gpu-xpu",
    disabled="OOM on stage-b XPU runner (server launch fails with --mem-fraction-static)",
)

if __name__ == "__main__":
    unittest.main()
