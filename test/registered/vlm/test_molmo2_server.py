import base64
import io
import unittest

import openai
from PIL import Image

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import ImageOpenAITestMixin

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")


def _solid_image_url(color, size) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestMolmo2Server(ImageOpenAITestMixin):
    model = "allenai/Molmo2-4B"

    def test_two_images_keep_their_order(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the color of Image 1 and of Image 2?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _solid_image_url((220, 20, 20), (300, 300))
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _solid_image_url((20, 40, 220), (500, 200))
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=32,
        )
        text = response.choices[0].message.content.lower()
        self.assertIn("red", text)
        self.assertIn("blue", text)
        self.assertLess(text.index("red"), text.index("blue"))


del ImageOpenAITestMixin


if __name__ == "__main__":
    unittest.main()
