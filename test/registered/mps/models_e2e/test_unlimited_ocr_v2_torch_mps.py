"""End-to-end Unlimited-OCR v2 (BF16) on the standard Torch ModelRunner on Apple MPS.

Covers the multimodal path (processor, SAM/CLIP encoder, DASH decoder with the
Metal kernels) and document splitting, against drawn pages with known text.
"""

import base64
import io
import json
import os
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_mps_ci(est_time=180, suite="stage-b-e2e-mps")

MODEL_PATH = os.environ.get("SGLANG_MPS_UOCR_MODEL", "baidu/Unlimited-OCR-2")
# Same no-repeat processor the model card's client attaches, pickled by reference.
NGRAM_PROCESSOR = json.dumps(
    {
        "callable": (
            b"csglang.srt.sampling.custom_logit_processor\n"
            b"DeepseekOCRNoRepeatNGramLogitProcessor\n."
        ).hex()
    }
)


def _page(title: str) -> str:
    """A white page with a title and numbered lines, as a PNG data URL."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 90), title, fill="black", font=ImageFont.load_default(size=56))
    body = ImageFont.load_default(size=34)
    for i in range(8):
        draw.text(
            (90, 220 + i * 60),
            f"{i + 1}. {title} line {i + 1}.",
            fill="black",
            font=body,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
class TestUnlimitedOcrV2TorchMps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "0"
        cls.process = popen_launch_server(
            try_cached_model(MODEL_PATH),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="mps",
            env=env,
            other_args=[
                "--chat-template",
                "unlimited-ocr",
                "--enable-custom-logit-processor",
                "--uocr-doc-split",
                "--max-total-tokens",
                "16384",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _ocr(self, titles: list[str]) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": _page(t)}} for t in titles
        ]
        text = "<PAGE><image>" * len(titles) + "Multi page parsing."
        content.append({"type": "text", "text": text})
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1024,
                "temperature": 0,
                "skip_special_tokens": False,
                "images_config": {"image_mode": "base"},
                "custom_logit_processor": NGRAM_PROCESSOR,
                "custom_params": {"ngram_size": 35, "window_size": 128},
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def test_single_page(self):
        text = self._ocr(["Apple Metal"])
        self.assertIn("Apple Metal line 1", text)
        self.assertIn("Apple Metal line 8", text)

    def test_document_split_reads_every_page(self):
        text = self._ocr(["First page", "Second page", "Third page"])
        for title in ("First page", "Second page", "Third page"):
            self.assertIn(f"{title} line 8", text)


if __name__ == "__main__":
    unittest.main()
