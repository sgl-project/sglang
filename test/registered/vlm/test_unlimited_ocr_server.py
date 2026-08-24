import base64
import io
import unittest

import openai
from PIL import Image, ImageDraw, ImageFont

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import TestOpenAIMLLMServerBase

register_cuda_ci(est_time=56, stage="base-b", runner_config="1-gpu-large")


class TestUnlimitedOCRServer(TestOpenAIMLLMServerBase):
    model = "baidu/Unlimited-OCR"
    trust_remote_code = False
    extra_args = [
        "--attention-backend=fa3",
        "--page-size=1",
        "--context-length=4096",
        "--max-total-tokens=4096",
        "--enable-custom-logit-processor",
        "--disable-radix-cache",
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
    def _make_ocr_image_url(cls) -> str:
        img = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((24, 24, 616, 336), outline="black", width=4)
        draw.text((72, 92), "SGLang OCR", fill="black", font=cls._font(56))
        draw.text((72, 180), "12345", fill="black", font=cls._font(72))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "document parsing."},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._make_ocr_image_url()},
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=64,
            extra_body={"images_config": {"image_mode": "gundam"}},
        )

        self.assertEqual(response.choices[0].message.role, "assistant")
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertIn("12345", text)
        self.assertIn("sglang", text.lower())
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)
        self.assertGreater(response.usage.total_tokens, 0)


del TestOpenAIMLLMServerBase


if __name__ == "__main__":
    unittest.main()
