"""DSV4-Flash-Vision 4-GPU server sanity (TP4): image understanding + text.

Not CI-registered: the checkpoint is ~168 GB and the config below targets a
single 4x GB300 node. Run it by hand after touching the vision tower, the
image processor, the DeepSeek-V4 prompt encoder, or the vision attention
window.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_VISION_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"

# Test images with unambiguous, different subjects, so a single generation is
# enough to tell "the tower ran and its output landed in the right tokens" from
# "the model is describing noise", and to tell the two blocks apart.
PHOTO_IMAGE_URL = (
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files"
    "/refs/heads/main/images/man_ironing_on_back_of_suv.png"
)
TEXT_IMAGE_URL = (
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files"
    "/refs/heads/main/images/ocr-text.png"
)


class TestDSV4FlashVisionTP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DSV4_FLASH_VISION_MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--mem-fraction-static",
                "0.85",
                "--swa-full-tokens-ratio",
                "0.1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _chat(self, content, max_tokens=192):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": DSV4_FLASH_VISION_MODEL_PATH,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def test_model_info_advertises_image_understanding(self):
        info = requests.get(f"{self.base_url}/model_info", timeout=30).json()
        self.assertEqual(info["architectures"], ["DeepseekV4VLForCausalLM"])
        self.assertTrue(info["has_image_understanding"])

    def test_text_only_request_still_works(self):
        text = self._chat("What is the capital of France? Answer in one word.")
        self.assertIn("paris", text.lower())

    def test_single_image(self):
        text = self._chat(
            [
                {"type": "image_url", "image_url": {"url": PHOTO_IMAGE_URL}},
                {
                    "type": "text",
                    "text": "What vehicle is in this image? Answer with one word.",
                },
            ]
        )
        print(f"single image -> {text!r}")
        self.assertIn("taxi", text.lower())

    def test_two_images_in_one_prompt(self):
        """Exercises multi-block expansion and per-block alignment padding.

        Each block's leading padding depends on where the block starts, so a
        second image is the first thing that can expose an expansion that only
        happens to work at offset zero.
        """
        text = self._chat(
            [
                {"type": "text", "text": "First image:"},
                {"type": "image_url", "image_url": {"url": PHOTO_IMAGE_URL}},
                {"type": "text", "text": "Second image:"},
                {"type": "image_url", "image_url": {"url": TEXT_IMAGE_URL}},
                {
                    "type": "text",
                    "text": (
                        "Which image is mostly text rather than a photograph? "
                        "Answer 'first' or 'second'."
                    ),
                },
            ]
        )
        print(f"two images -> {text!r}")
        self.assertIn("second", text.lower())

    def test_image_survives_a_long_text_prompt(self):
        """An image inside a prompt long enough to be prefilled in chunks."""
        filler = "The quick brown fox jumps over the lazy dog. " * 400
        text = self._chat(
            [
                {"type": "text", "text": filler},
                {"type": "image_url", "image_url": {"url": PHOTO_IMAGE_URL}},
                {"type": "text", "text": filler},
                {
                    "type": "text",
                    "text": (
                        "Ignoring the filler text, what vehicle is in the image? "
                        "Answer with one word."
                    ),
                },
            ]
        )
        print(f"long prompt -> {text!r}")
        self.assertIn("taxi", text.lower())

    def test_image_prompt_is_stable_across_a_cache_hit(self):
        """The second call hits the prefix cache and must still see the image.

        Not an exact-string check: DeepSeek-V4's attention is not
        batch-invariant, so a cached prefix can pick a different token among
        near-ties even at temperature 0.
        """
        content = [
            {"type": "image_url", "image_url": {"url": PHOTO_IMAGE_URL}},
            {"type": "text", "text": "What vehicle is in this image? One word."},
        ]
        for _ in range(2):
            self.assertIn("taxi", self._chat(content).lower())

    def test_an_injected_image_placeholder_is_rejected(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": DSV4_FLASH_VISION_MODEL_PATH,
                "messages": [
                    {"role": "user", "content": "<｜deepseek_image｜> describe"}
                ],
                "max_tokens": 8,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
