"""GLM image chats through the native Rust server and real vision encoder.

Reversing tool-result arrival must preserve the images' association with call
ids. Splitting text into adjacent parts must preserve the prompt after an image
turn. CPU fixtures check rendering and pixels separately; these requests also
exercise token expansion, TP shared-memory handoff, M-RoPE, and inference.
"""

import base64
import copy
import io
import os
import re
import tempfile
import unittest
from pathlib import Path

import requests
from PIL import Image

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

# The checkpoint is 320B FP8; use its existing four-B200 serving recipe.
register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-b200")


def image_part(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


class TestRustGlm53FlashImages(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model("zai-org/GLM-5.3-Flash")
        cls.log_dir = tempfile.TemporaryDirectory()
        cls.log_paths = [Path(cls.log_dir.name) / name for name in ("out", "err")]
        cls.logs = tuple(path.open("w") for path in cls.log_paths)
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=3600,
            other_args=[
                "--tp-size",
                "4",
                "--enable-multimodal",
                "--image-processor-backend",
                "pil",
                "--dsa-prefill-backend",
                "trtllm",
                "--dsa-decode-backend",
                "trtllm",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
                "--mem-fraction-static",
                "0.8",
            ],
            env={**os.environ, "SGLANG_RUST_SERVER": "1"},
            return_stdout_stderr=cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()
        if hasattr(cls, "log_dir"):
            cls.log_dir.cleanup()

    def chat(self, messages):
        self.assertIn(
            "Rust MM pipeline enabled",
            "\n".join(path.read_text() for path in self.log_paths),
        )
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
                "reasoning_effort": "low",
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["choices"][0]["finish_reason"], "stop", result)
        return result

    def colors(self, result):
        answer = result["choices"][0]["message"]["content"]
        self.assertTrue(answer, result)
        return re.findall(r"\b(red|blue)\b", answer.lower())

    def test_tool_result_arrival_preserves_image_association(self):
        messages = [
            {"role": "user", "content": "Fetch images A and B."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": "fetch_" + name, "arguments": "{}"},
                    }
                    for name in ("A", "B")
                ],
            },
            {"role": "tool", "tool_call_id": "A", "content": [image_part("red")]},
            {"role": "tool", "tool_call_id": "B", "content": [image_part("blue")]},
            {
                "role": "user",
                "content": "Name the color of image A, then image B. Reply with only two color words.",
            },
        ]
        ordered = self.chat(messages)
        messages[2:4] = reversed(messages[2:4])
        reversed_results = self.chat(messages)
        for result in (ordered, reversed_results):
            self.assertEqual(self.colors(result), ["red", "blue"], result)
        self.assertEqual(
            ordered["usage"]["prompt_tokens"],
            reversed_results["usage"]["prompt_tokens"],
        )

    def test_text_parts_after_image_preserve_prompt(self):
        messages = [
            {
                "role": "user",
                "content": [
                    image_part("red"),
                    {"type": "text", "text": "Remember this image."},
                ],
            },
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Repeat the image's color as one word."},
        ]
        flat = self.chat(messages)
        parts = copy.deepcopy(messages)
        parts[-1]["content"] = [
            {"type": "text", "text": "Re"},
            {"type": "text", "text": "peat the image's color as one word."},
        ]
        split = self.chat(parts)
        self.assertEqual(
            flat["usage"]["prompt_tokens"], split["usage"]["prompt_tokens"]
        )
        for result in (flat, split):
            self.assertEqual(self.colors(result), ["red"], result)


if __name__ == "__main__":
    unittest.main()
