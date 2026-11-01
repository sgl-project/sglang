"""
python3 -m unittest test_unlimited_ocr.py
"""

import base64
import io
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw, ImageFont

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=180, suite="stage-b-test-1-gpu-xpu")

# Every layer of Unlimited-OCR is sliding-window (window 128), so it runs on
# PureSWATokenToKVPoolAllocator, which is page_size 1 only.
SLIDING_WINDOW = 128
NUM_LINES = 40


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


def _dense_page_url() -> str:
    """A letter-size page large enough to take the tiling path, with one
    numbered line per row so the transcription is long and checkable."""
    w, h = 1700, 2200
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    size = h // (NUM_LINES + 8)
    font = _font(size)
    draw.text((w // 20, size), "REPORT 0001", fill="black", font=font)
    for i in range(NUM_LINES):
        draw.text(
            (w // 20, int(size * (2.4 + i * 1.25))),
            f"{i + 1:02d}. item {i} amount {i * 13 % 97}.50 USD",
            fill="black",
            font=font,
        )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class TestUnlimitedOCRPageSize1(CustomTestCase):
    """Pure-SWA attention at page_size 1 on intel_xpu: the server must start
    without rewriting page_size, and decode must stay correct after it
    crosses the sliding window."""

    @classmethod
    def setUpClass(cls):
        cls.model = "baidu/Unlimited-OCR"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.image_url = _dense_page_url()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "xpu",
                "--attention-backend",
                "intel_xpu",
                "--page-size",
                "1",
                "--mem-fraction-static",
                "0.7",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=30)
            except Exception:
                kill_process_tree(cls.process.pid)

    def _parse(self, max_tokens: int) -> dict:
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "document parsing."},
                            {"type": "image_url", "image_url": {"url": self.image_url}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "images_config": {"image_mode": "gundam"},
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def test_page_size_is_not_rewritten(self):
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(info["page_size"], 1)

    def test_decode_past_sliding_window(self):
        out = self._parse(max_tokens=512)
        text = out["choices"][0]["message"]["content"]
        completion_tokens = out["usage"]["completion_tokens"]
        self.assertGreater(completion_tokens, 2 * SLIDING_WINDOW)
        # The first lines are read inside the window; the later ones only after
        # decode has run well past it, so they check the windowed path.
        self.assertIn("REPORT", text)
        self.assertIn("01. item 0", text)
        self.assertIn("08. item 7", text)

    def test_concurrent_requests(self):
        # Several requests share page_size 1 KV allocation and window eviction.
        # Same-sized images, since batched tiling needs equal crop counts.
        with ThreadPoolExecutor(max_workers=4) as pool:
            outs = list(pool.map(lambda _: self._parse(max_tokens=256), range(4)))
        for out in outs:
            text = out["choices"][0]["message"]["content"]
            self.assertIn("01. item 0", text)
            self.assertIn("05. item 4", text)


if __name__ == "__main__":
    unittest.main()
