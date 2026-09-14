"""End-to-end OpenAI-API coverage for PaddleOCR-VL.

The vision tower encodes a whole batch as one packed tensor, so the test drives
a single-image request plus several concurrent requests with differently sized
images — the shape that makes the scheduler hand several images to one ViT
forward. Bit-exactness of the packing itself is pinned on CPU by
`test/registered/unit/models/test_paddleocr_vl_vision.py`.
"""

import base64
import io
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai
from PIL import Image, ImageDraw, ImageFont

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import TestOpenAIMLLMServerBase

register_cuda_ci(est_time=63, stage="base-b", runner_config="1-gpu-large")


class TestPaddleOCRVLServer(TestOpenAIMLLMServerBase):
    model = "PaddlePaddle/PaddleOCR-VL"
    extra_args = [
        "--context-length=8192",
        "--mem-fraction-static=0.7",
        "--cuda-graph-max-bs-decode=4",
    ]

    @staticmethod
    def _font(size: int):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
        return ImageFont.load_default()

    @classmethod
    def _make_ocr_image_url(cls, text: str, size=(640, 360)) -> str:
        width, height = size
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((16, 16, width - 16, height - 16), outline="black", width=4)
        font_size = height // 6
        font = cls._font(font_size)
        text_width = draw.textbbox((0, 0), text, font=font)[2]
        if text_width > width - 96:
            font = cls._font((width - 96) * font_size // text_width)
        draw.text((48, height // 3), text, fill="black", font=font)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _ocr(self, client, image_url: str, max_tokens: int = 64) -> str:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR:"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)
        return response.choices[0].message.content

    def test_single_image_ocr(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        text = self._ocr(client, self._make_ocr_image_url("SGLANG 12345"))

        self.assertIsInstance(text, str)
        self.assertIn("12345", text)
        self.assertIn("sglang", text.lower())

    def test_concurrent_requests_batch_the_vision_tower(self):
        """Different image sizes in flight at once must not bleed across images."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        cases = [
            ("ALPHA 111", (640, 360)),
            ("BRAVO 222", (800, 320)),
            ("CHARLIE 333", (512, 512)),
            ("DELTA 444", (960, 288)),
        ]
        urls = [self._make_ocr_image_url(text, size) for text, size in cases]

        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            results = list(pool.map(lambda url: self._ocr(client, url), urls))

        for (expected_text, _), actual in zip(cases, results):
            word, digits = expected_text.split()
            self.assertIn(digits, actual, f"{expected_text!r} -> {actual!r}")
            self.assertIn(
                word.lower(), actual.lower(), f"{expected_text!r} -> {actual!r}"
            )


del TestOpenAIMLLMServerBase


if __name__ == "__main__":
    unittest.main()
