"""Native Jina OCR serving smoke test (requires a CUDA GPU).

Run: python -m unittest test.registered.vlm.test_jina_ocr_server
The target uses the existing DeepSeek-OCR engine; FastMTP is not enabled.
"""

import base64
import io
import unittest

import openai
from PIL import Image, ImageDraw, ImageFont

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import TestOpenAIMLLMServerBase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")


class TestJinaOCRServer(TestOpenAIMLLMServerBase):
    model = "jinaai/jina-ocr-v1"
    trust_remote_code = False
    extra_args = [
        "--context-length=4096",
        "--mem-fraction-static=0.7",
        "--cuda-graph-max-bs-decode=4",
    ]

    def test_ocr(self):
        image = Image.new("RGB", (1920, 1920), "white")
        ImageDraw.Draw(image).text(
            (40, 80), "CHARLIE 333", fill="black", font=ImageFont.load_default(size=48)
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        with openai.Client(api_key=self.api_key, base_url=self.base_url) as client:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe the provided document image into a clean "
                                    "Markdown format, preserving the natural reading order."
                                ),
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=64,
            )
        self.assertIn("CHARLIE 333", response.choices[0].message.content or "")


del TestOpenAIMLLMServerBase

if __name__ == "__main__":
    unittest.main()
