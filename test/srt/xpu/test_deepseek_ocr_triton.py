"""
python3 -m unittest test_deepseek_ocr_triton.py
"""

import os
import unittest
from pathlib import Path

import test_deepseek_ocr as deepseek_ocr
from transformers import AutoTokenizer

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestDeepSeekOCRTriton(deepseek_ocr.TestDeepSeekOCR):
    @classmethod
    def setUpClass(cls):
        cls._cleanup_xpu_memory()
        cls.model = "deepseek-ai/DeepSeek-OCR"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, use_fast=False, trust_remote_code=True
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.image_path = str(
            (Path(__file__).resolve().parents[3] / "examples/assets/example_image.png")
        )
        if not os.path.exists(cls.image_path):
            raise FileNotFoundError(f"Image not found: {cls.image_path}")
        cls.common_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
        ]
        os.environ["SGLANG_USE_SGL_XPU"] = "0"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )


if __name__ == "__main__":
    unittest.main()
