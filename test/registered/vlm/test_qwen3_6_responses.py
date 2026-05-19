"""End-to-end /v1/responses regression test on a multimodal Qwen3.6 model.

Covers https://github.com/sgl-project/sglang/issues/25593: before the fix,
POST /v1/responses on a multimodal model returned HTTP 400 with
"input_ids should be a list of lists for batch processing." because the
multimodal branch of OpenAIServingResponses._make_request put a rendered
chat-template *string* into GenerateReqInput.input_ids instead of .text.

There were no e2e tests that combined a multimodal model with /v1/responses
before — VLM e2e tests only covered /v1/chat/completions, and the existing
/v1/responses suite (TestOpenAIServerv1Responses) ran on a text-only model.
This file plugs that gap by booting a real multimodal model and exercising
the simplest input format the issue reporter used: ``input: "<string>"``.

This test deliberately uses text-only input even though the model is
multimodal — that's the exact scenario the issue describes, and it's the
narrowest regression we can write that still goes through the multimodal
branch of _make_request (selected by ``model_config.is_multimodal``, not by
the request payload). Multimodal-input coverage on /v1/responses is a
separate bug (Responses-API content schema is not converted to Chat
Completions schema before reuse) and is not in scope here.

Usage:
    python3 test/registered/vlm/test_qwen3_6_responses.py
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-large")


class TestQwen3_6Responses(CustomTestCase):
    """Boot Qwen3.6-27B and hit /v1/responses with text input."""

    model = "Qwen/Qwen3.6-27B"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--enable-multimodal",
                "--cuda-graph-max-bs=4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_responses_text_only(self):
        """Exact curl from issue #25593 — before the fix this returned 400."""
        r = requests.post(
            f"{self.base_url}/v1/responses",
            headers={"Content-Type": "application/json"},
            json={"model": self.model, "input": "hello"},
            timeout=120,
        )
        self.assertEqual(r.status_code, 200, msg=r.text)
        body = r.json()
        self.assertEqual(body.get("object"), "response")
        self.assertEqual(body.get("status"), "completed")
        # Sanity: the assistant produced some text.
        texts = [
            c["text"]
            for item in body.get("output", [])
            if item.get("type") == "message"
            for c in item.get("content", [])
            if c.get("type") == "output_text" and c.get("text")
        ]
        self.assertTrue("".join(texts).strip(), msg=body)


if __name__ == "__main__":
    unittest.main()
