"""
Gemma 4 E2B-it: simple text Q&A on XPU (OpenAI /v1), same shape as
``test_deepseek_coder_v2_lite_instruct.py``.

Model card: https://huggingface.co/google/gemma-4-E2B-it

  - XPU test runs when Intel XPU is available.

Run from test/srt::

  python3 -m unittest xpu.test_gemma_4_e2b.TestGemma4E2BXPU.test_simple_qa

Full-model and text-only launches verify that the model boots on XPU,
preserves processor token IDs, and rejects media without its encoders.

Server is started with ``sglang serve`` (``--model-impl sglang``).
"""

from __future__ import annotations

import unittest

import openai
import requests
import torch

from sglang.srt.utils.common import is_xpu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer_from_processor,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.vlm_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
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
    server_args = XPU_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        _empty_xpu_cache()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            api_key=cls.api_key,
            other_args=list(cls.server_args),
            device="xpu",
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)
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


class TestGemma4E2BTextOnlyXPU(TestGemma4E2BXPU):
    """The text-only launch must omit media capability and reject media inputs."""

    server_args = XPU_SERVER_ARGS + [
        "--language-model-only",
        "--cuda-graph-backend-decode",
        "full",
    ]

    def test_preserves_processor_token_ids(self):
        tokenizer = get_tokenizer_from_processor(
            get_processor(self.model, trust_remote_code=True)
        )
        prompt = tokenizer.apply_chat_template(
            _simple_text_messages(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = requests.post(
            f"{self.base_url}/tokenize",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"prompt": prompt, "add_special_tokens": True},
            timeout=30,
        )
        response.raise_for_status()
        self.assertEqual(response.json()["tokens"], tokenizer.encode(prompt))

    def test_rejects_media_without_encoders(self):
        url = self.base_url.removesuffix("/v1")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        info = requests.get(f"{url}/model_info", headers=headers, timeout=30)
        info.raise_for_status()
        self.assertFalse(info.json()["has_image_understanding"])
        self.assertFalse(info.json()["has_audio_understanding"])
        response = requests.post(
            f"{url}/generate",
            headers=headers,
            json={
                "text": "Describe this image.",
                "image_data": "data:image/png;base64,AA==",
                "sampling_params": {"max_new_tokens": 1},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("--language-model-only", response.text)


# Full-model and text-only launches.
register_xpu_ci(
    est_time=360,
    suite="nightly-xpu-kernel-main-1-gpu",
    nightly=True,
)
register_xpu_ci(
    est_time=360,
    suite="nightly-xpu-kernel-wheel-1-gpu",
    nightly=True,
)

if __name__ == "__main__":
    unittest.main()
