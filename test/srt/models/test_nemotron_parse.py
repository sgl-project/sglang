r"""
Test Nemotron Parse model via /generate endpoint.

Requires a running server:
    python -m sglang.launch_server \
        --model nvidia/NVIDIA-Nemotron-Parse-v1.1 \
        --trust-remote-code \
        --port 29999

Usage:
    pip install albumentations open_clip_torch timm
    python test/srt/models/test_nemotron_parse.py
    python test/srt/models/test_nemotron_parse.py --port 29999

Expected successful output (example):
    ....
    ----------------------------------------------------------------------
    Ran 4 tests in <time>s

    OK
    Found 10 bbox coords, 5 class labels
    Classes: ['<class_Page-header>', '<class_Text>', '<class_Section-header>', '<class_Text>', '<class_Table>']
    Detected classes: {'Page-header', 'Text', 'Table', 'Section-header'}
    Table extraction OK
    Extracted text:
    <x_0.0293><y_0.0148>Document<x_0.0723><y_0.0187><class_Page-header>

    <x_0.0293><y_0.0391>This is a paragraph of text in a simple document for testing OCR capabilities.<x_0.1338><y_0.0539><class_Text>

    <x_0.0293><y_0.0781>Section 2: Results<x_0.0781><y_0.0828><class_Section-header>

    <x_0.0293><y_0.0977>The results show improvement<x_0.1162><y_0.1023><class_Text>

    <x_0.0293><y_0.1313>\begin{tabular}{cc}
    Column A & Column B \\
    1.0 & 2.0 \\
    3.0 & 4.0 \\
    \end{tabular}<x_0.2422><y_0.175><class_Table>

Notes:
    - Runtime and class-set ordering may vary.
    - Output text may differ slightly across model versions.
"""

import argparse
import base64
import io
import json
import re
import unittest
import urllib.request

from PIL import Image, ImageDraw

DEFAULT_PORT = 29999
DEFAULT_HOST = "127.0.0.1"

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"

SAMPLING_PARAMS = {
    "max_new_tokens": 4000,
    "temperature": 0.0,
    "repetition_penalty": 1.1,
    "top_k": 1,
}


def make_test_image():
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)

    draw.text((50, 30), "Document Title", fill="black")
    draw.text((50, 80), "This is a paragraph of text in a simple", fill="black")
    draw.text((50, 100), "document for testing OCR capabilities.", fill="black")
    draw.text((50, 160), "Section 2: Results", fill="black")
    draw.text((50, 200), "The results show improvement.", fill="black")

    for i in range(4):
        draw.line([(50, 270 + i * 30), (400, 270 + i * 30)], fill="black")
    draw.line([(50, 270), (50, 360)], fill="black")
    draw.line([(200, 270), (200, 360)], fill="black")
    draw.line([(400, 270), (400, 360)], fill="black")
    draw.text((80, 275), "Column A", fill="black")
    draw.text((260, 275), "Column B", fill="black")
    draw.text((100, 305), "1.0", fill="black")
    draw.text((290, 305), "2.0", fill="black")
    draw.text((100, 335), "3.0", fill="black")
    draw.text((290, 335), "4.0", fill="black")

    return img


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate(host, port, image_b64, prompt=PROMPT, sampling_params=None):
    if sampling_params is None:
        sampling_params = SAMPLING_PARAMS

    payload = {
        "text": prompt,
        "image_data": [f"data:image/png;base64,{image_b64}"],
        "sampling_params": sampling_params,
    }

    req = urllib.request.Request(
        f"http://{host}:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


class TestNemotronParse(unittest.TestCase):
    host = DEFAULT_HOST
    port = DEFAULT_PORT

    @classmethod
    def setUpClass(cls):
        img_b64 = image_to_base64(make_test_image())
        cls.result = generate(cls.host, cls.port, img_b64)
        cls.text = cls.result["text"]
        cls.meta = cls.result["meta_info"]

    def test_basic_ocr(self):
        self.assertEqual(self.meta["finish_reason"]["type"], "stop")

        bbox_pattern = r"<x_[\d.]+><y_[\d.]+>"
        bboxes = re.findall(bbox_pattern, self.text)
        self.assertGreater(len(bboxes), 0, "No bounding boxes found in output")

        class_pattern = r"<class_[\w-]+>"
        classes = re.findall(class_pattern, self.text)
        self.assertGreater(len(classes), 0, "No class labels found in output")

        print(f"Found {len(bboxes)} bbox coords, {len(classes)} class labels")
        print(f"Classes: {classes}")

    def test_text_extraction(self):
        self.assertIn("Document", self.text)
        self.assertIn("Section", self.text)
        self.assertIn("Results", self.text)

        print(f"Extracted text:\n{self.text}")

    def test_table_extraction(self):
        self.assertIn("<class_Table>", self.text)
        self.assertIn("Column A", self.text)
        self.assertIn("Column B", self.text)

        print("Table extraction OK")

    def test_class_types(self):
        classes = re.findall(r"<class_([\w-]+)>", self.text)

        self.assertIn("Text", classes)
        self.assertIn("Table", classes)

        print(f"Detected classes: {sorted(set(classes))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, remaining = parser.parse_known_args()

    TestNemotronParse.host = args.host
    TestNemotronParse.port = args.port

    unittest.main(argv=["test_nemotron_parse"] + remaining)
