"""Per-commit end-to-end server test for Inkling (hybrid SWA/sconv attention + MoE + multimodal towers).

Boots a small ``thinkingmachines/Inkling`` checkpoint (the ``test`` revision, a
full-architecture but shrunken model) via ``popen_launch_server`` and exercises
the Inkling code paths on every PR: generation, the multimodal (vision) path,
the ``inkling`` reasoning parser, and UnifiedRadixTree cache consistency.

The checkpoint is undertrained, so these guard that the code paths boot and stay
numerically correct -- not answer quality; there is no accuracy gate (full-model
gsm8k accuracy lives in the 8-GPU nightly).
"""

import base64
import io
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci

# Aliased so pytest does not collect the imported `test_`-prefixed helper as a test.
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

# Defaults to the HF `test` revision; override MODEL/REVISION to point at a
# local checkpoint. Empty REVISION drops the flag (for local paths).
_MODEL_PATH = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")


def _small_image_data_uri():
    from PIL import Image

    im = Image.new("RGB", (64, 64), (200, 60, 60))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestInklingServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        # `enable_mamba_extra_buffer` is asserted by the Inkling model, and its
        # default attention backend is unsupported, so fa4 is required. KV pool
        # size follows mem_fraction_static, so it scales down on smaller GPUs.
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "fa4",
            "--page-size",
            "128",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--swa-full-tokens-ratio",
            "0.1",
            "--mamba-full-memory-ratio",
            "0.1",
            "--mem-fraction-static",
            "0.5",
            "--reasoning-parser",
            "inkling",
            "--tool-call-parser",
            "inkling",
            "--enable-multimodal",
        ]
        if _MODEL_REVISION:
            other_args += ["--revision", _MODEL_REVISION]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def _chat(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 48,
        }
        payload.update(kwargs)
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=120
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_generation_basic(self):
        """Each prompt must return a non-empty completion."""
        prompts = [
            "The capital of France is",
            "1 + 2 + 3 + 4 + 5 =",
            "Write a haiku about silicon:",
        ]
        for prompt in prompts:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
                },
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertIn("text", data, data)
            self.assertGreater(len(data["text"].strip()), 0, data)

    def test_reasoning_parser_separates_thinking(self):
        """`thinking=True` must route the chain-of-thought into
        `reasoning_content`, exercising the `inkling` reasoning parser."""
        data = self._chat(
            [{"role": "user", "content": "What is 17 * 24? Think step by step."}],
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"thinking": True}},
        )
        msg = data["choices"][0]["message"]
        self.assertTrue(
            (msg.get("reasoning_content") or "").strip(),
            f"expected non-empty reasoning_content, got {msg}",
        )

    def test_multimodal_image_is_consumed(self):
        """An image input must reach the vision tower (more prompt tokens than
        the text-only turn) and produce output, guarding the mm processor and
        placeholder handling against crashes/regressions."""
        text_only = self._chat([{"role": "user", "content": "Describe."}])
        with_image = self._chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe."},
                        {
                            "type": "image_url",
                            "image_url": {"url": _small_image_data_uri()},
                        },
                    ],
                }
            ],
            max_tokens=48,
            extra_body={"chat_template_kwargs": {"thinking": True}},
        )
        self.assertGreater(
            with_image["usage"]["prompt_tokens"],
            text_only["usage"]["prompt_tokens"],
            "image did not add prompt tokens -- vision path not exercised",
        )
        msg = with_image["choices"][0]["message"]
        produced = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        self.assertTrue(produced.strip(), f"empty completion for image input: {msg}")

    def test_unified_radix_cache_logprob_consistency(self):
        """UnifiedRadixTree correctness: prefill vs decode logprobs over real
        multi-turn prompts must agree (avg KL below threshold). A hybrid
        sconv/mamba/SWA state-cache bug shows up as a large KL. Numerical, so it
        is meaningful even on an undertrained checkpoint."""
        assert_logprobs_match(
            self.base_url,
            {self.model: {"kl_div": 1e-2}},
            self.model,
            max_samples=4,
            max_new_tokens=256,
            trust_remote_code=True,
        )


if __name__ == "__main__":
    unittest.main()
