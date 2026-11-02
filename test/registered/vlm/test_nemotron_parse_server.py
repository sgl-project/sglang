import base64
import io
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw, ImageFont

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-large")

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


class TestNemotronParseServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            os.environ.get(
                "SGLANG_TEST_NEMOTRON_PARSE_MODEL", "nvidia/Nemotron-3.5-Parse"
            ),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                "nemotron-parse",
                "--revision",
                "4035ffb1d8f8bafcdb867dfaf6f7514f101464f9",
                "--trust-remote-code",
                "--context-length",
                "16384",
                "--max-running-requests",
                "4",
                "--max-total-tokens",
                "16384",
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @staticmethod
    def make_page(text, size):
        image = Image.new("RGB", size, "white")
        ImageDraw.Draw(image).text(
            (40, 40), text, font=ImageFont.load_default(size=32), fill="black"
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def parse_page(self, page):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": PROMPT,
                "image_data": page,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.1,
                    "skip_special_tokens": False,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["meta_info"]["finish_reason"]["type"], "stop")
        blocks = re.findall(
            r"<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>",
            result["text"],
            re.DOTALL,
        )
        self.assertTrue(blocks, result["text"])
        for x1, y1, text, x2, y2, class_name in blocks:
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            self.assertTrue(0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1)
        return result["text"]

    def test_generate_and_chat_document_parsing(self):
        page = self.make_page("Paris is the capital of France.", (850, 1050))
        output = self.parse_page(page)
        self.assertIn("Paris is the capital of France.", output)
        self.assertIn("<class_Text>", output)
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": "nemotron-parse",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": page}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 256,
                "skip_special_tokens": False,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        text = result["choices"][0]["message"]["content"]
        self.assertIn("Paris is the capital of France.", text)
        self.assertIn("<class_Text>", text)

    def test_concurrent_pages_with_different_sizes(self):
        cases = [
            ("ALPHA 111", (640, 360)),
            ("BRAVO 222", (800, 320)),
            ("CHARLIE 333", (512, 512)),
            ("DELTA 444", (960, 288)),
        ]
        pages = [self.make_page(text, size) for text, size in cases]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(self.parse_page, pages))
        for (expected, _), output in zip(cases, outputs, strict=True):
            self.assertIn(expected, output)
            for other, _ in cases:
                if other != expected:
                    self.assertNotIn(other, output)


if __name__ == "__main__":
    unittest.main()
