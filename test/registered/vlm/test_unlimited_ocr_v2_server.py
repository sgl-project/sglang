import base64
import io
import unittest

import openai
from PIL import Image, ImageDraw, ImageFont

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import TestOpenAIMLLMServerBase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

PAGE = "<PAGE>"
# One distinct number per page of the multi-page document, in page order.
PAGE_NUMBERS = ("31415", "27182", "16180")


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


def _image_part(img: Image.Image) -> dict:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def _ocr_image() -> Image.Image:
    img = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 24, 616, 336), outline="black", width=4)
    draw.text((72, 92), "SGLang OCR", fill="black", font=_font(56))
    draw.text((72, 180), "12345", fill="black", font=_font(72))
    return img


def _document_page(number: str) -> Image.Image:
    img = Image.new("RGB", (768, 1024), "white")
    draw = ImageDraw.Draw(img)
    draw.text((80, 120), "Quarterly Report", fill="black", font=_font(56))
    draw.text((80, 320), f"Account {number}", fill="black", font=_font(72))
    return img


def _document_content(pages) -> list:
    """The model card's prompt: images first, then one marker run per page."""
    text = PAGE + "<image>"
    prompt = text * len(pages) + "Multi page parsing."
    return [_image_part(page) for page in pages] + [{"type": "text", "text": prompt}]


class TestUnlimitedOCRV2Server(TestOpenAIMLLMServerBase):
    model = "baidu/Unlimited-OCR-2"
    trust_remote_code = False
    extra_args = [
        "--revision=5632329a32e103d5f1add270681e0f5c1233c6e4",
        "--chat-template=unlimited-ocr",
        "--enable-streaming-session",
        # Small per-page cap and one page per base-mode window, so the
        # three-page document splits into three windows that reuse KV.
        "--uocr-max-page-tokens=512",
        "--uocr-doc-split",
        "--uocr-doc-page-window=1",
        # Workaround: eager decode degenerates into repeated numbers on this
        # checkpoint; drop once it matches the decode CUDA graph path.
        "--cuda-graph-backend-decode=full",
        "--cuda-graph-max-bs-decode=4",
        "--mem-fraction-static=0.7",
    ]

    def _chat(self, content, *, max_tokens, crop_mode):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        return client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={
                "images_config": {"crop_mode": crop_mode},
                "skip_special_tokens": False,
            },
        )

    def test_single_image_chat_completion(self):
        response = self._chat(
            _document_content([_ocr_image()]), max_tokens=256, crop_mode=True
        )

        self.assertEqual(response.choices[0].message.role, "assistant")
        text = response.choices[0].message.content
        self.assertIn("12345", text)
        self.assertIn("sglang", text.lower())
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)

    def test_external_processors_remain_disabled(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        with self.assertRaisesRegex(
            openai.BadRequestError, "--enable-custom-logit-processor"
        ):
            client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                extra_body={"custom_logit_processor": "external-processor"},
            )

    def test_split_document_returns_every_page_in_order(self):
        pages = [_document_page(number) for number in PAGE_NUMBERS]
        response = self._chat(
            _document_content(pages), max_tokens=512 * len(pages), crop_mode=False
        )

        metadata = response.model_dump()["metadata"]
        self.assertEqual(metadata["doc_split_batches"], len(pages))
        self.assertTrue(metadata["doc_split_complete"], metadata)
        # Every window after the first spliced its predecessor's KV tail.
        self.assertEqual(metadata["doc_split_reused_batches"], [1, 2])
        bodies = response.choices[0].message.content.split(PAGE)[1:]
        self.assertEqual(len(bodies), len(pages), bodies)
        for number, body in zip(PAGE_NUMBERS, bodies):
            with self.subTest(number=number):
                self.assertIn(number, body)


del TestOpenAIMLLMServerBase


if __name__ == "__main__":
    unittest.main()
