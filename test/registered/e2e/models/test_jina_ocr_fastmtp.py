import os
import tempfile
import unittest

import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="nightly", runner_config="1-gpu-large")

MODEL = "jinaai/jina-ocr-v1"
OCR_PROMPT = (
    "Transcribe the provided document image into a clean Markdown format, "
    "preserving the natural reading order."
)
DOC_LINES = [
    "Quarterly Report",
    "Revenue grew by 12 percent compared to last year.",
    "Operating costs remained flat across all regions.",
    "The board approved a new research budget.",
]
COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.7",
]
SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
MAX_NEW_TOKENS = 256
# Arbitrary floor well below FastMTP's K=3 ceiling of 4; a draft with missing
# or misrouted weights sits near 1.0.
ACCEPT_LENGTH_THRES = 1.8


def _render_document(path: str) -> None:
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    for i, line in enumerate(DOC_LINES):
        draw.text((60, 80 + i * 70), line, fill="black", font=font)
    image.save(path)


class TestJinaOCRFastMTP(CustomTestCase):
    """FastMTP speculative decoding must reproduce the greedy target output."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.image_path = os.path.join(cls.tmpdir.name, "doc.png")
        _render_document(cls.image_path)

        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        cls.prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        cls.base_out = cls._run_server(other_args=COMMON_ARGS)
        cls.spec_out = cls._run_server(other_args=COMMON_ARGS + SPEC_ARGS)

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        cls.tmpdir.cleanup()

    @classmethod
    def _run_server(cls, other_args):
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            response = requests.post(
                cls.base_url + "/generate",
                json={
                    "text": cls.prompt,
                    "image_data": cls.image_path,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json()
        finally:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None

    def test_target_transcribes_document(self):
        text = self.base_out["text"]
        for line in DOC_LINES:
            self.assertIn(line, text)

    def test_spec_output_matches_target(self):
        self.assertEqual(self.spec_out["text"], self.base_out["text"])

    def test_fastmtp_accept_length(self):
        accept_length = self.spec_out["meta_info"]["spec_accept_length"]
        print(f"FastMTP accept length: {accept_length:.2f}")
        self.assertGreater(accept_length, ACCEPT_LENGTH_THRES)


if __name__ == "__main__":
    unittest.main(verbosity=3)
