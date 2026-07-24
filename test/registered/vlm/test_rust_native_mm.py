"""E2E: Rust tokenizer-manager multimodal path (``SGLANG_RUST_SERVER=1``).

Covers what the CPU parity units structurally cannot: the sidecar handoff
(``take_native_mm`` wrapping Rust-owned buffers), the drain ordering, live
fallback to the Python mm_processor, and the Rust-side text tokenization of
multimodal prompts. ``TestRustServerPythonMm`` additionally pins the
``SGLANG_DISABLE_NATIVE_MM`` escape hatch.
"""

import base64
import importlib.util
import io
import os
import unittest

import numpy as np
import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

IMAGE_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def chat_prompt(question, image_count=1):
    return (
        f"<|im_start|>user\n{VISION_BLOCK * image_count}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def solid_image_data_url(fmt):
    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)).save(
        buffer, format=fmt
    )
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{encoded}"


@unittest.skipIf(
    importlib.util.find_spec("sglang_server") is None,
    "sglang_server wheel not installed (e.g. AMD suite)",
)
class TestRustServerNativeMm(CustomTestCase):
    env = {"SGLANG_RUST_SERVER": "1"}

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-multimodal", "--mem-fraction-static", "0.8"],
            env={**os.environ, **cls.env},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def generate(self, prompt, image_data):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompt,
                "image_data": image_data,
                "sampling_params": {"temperature": 0, "max_new_tokens": 48},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["text"].lower()

    def test_single_image_url(self):
        text = self.generate(
            chat_prompt("Describe this image in one sentence."), [IMAGE_URL]
        )
        keywords = ("iron", "man", "taxi", "cab", "car", "suv", "street")
        self.assertTrue(any(w in text for w in keywords), text)

    def test_two_images(self):
        red = solid_image_data_url("PNG")
        text = self.generate(
            chat_prompt("What color is the second image?", image_count=2),
            [IMAGE_URL, red],
        )
        self.assertIn("red", text)

    def test_unsupported_format_falls_back(self):
        # GIF is not decodable natively; the request must still succeed via
        # the Python mm_processor fallback.
        text = self.generate(
            chat_prompt("What color is this image?"), [solid_image_data_url("GIF")]
        )
        self.assertIn("red", text)

    def test_corrupt_image_is_rejected(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": chat_prompt("Describe this image."),
                "image_data": ["data:image/png;base64,aW52YWxpZA=="],
                "sampling_params": {"max_new_tokens": 8},
            },
        )
        # The rust server surfaces mm-processor failures as Error::Encode
        # (500); the request must be rejected without killing the server.
        self.assertIn(response.status_code, (400, 500), response.text)


class TestRustServerPythonMm(TestRustServerNativeMm):
    """Same rust server, Python mm_processor path (native MM disabled)."""

    env = {"SGLANG_RUST_SERVER": "1", "SGLANG_DISABLE_NATIVE_MM": "1"}


if __name__ == "__main__":
    unittest.main(verbosity=3)
