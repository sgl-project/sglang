"""
python3 -m unittest test_deepseek_ocr_triton.py
"""

import os
import unittest
from pathlib import Path

from test_deepseek_ocr import TestDeepSeekOCR

from sglang.srt.utils.hf_transformers import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


# TODO: Temporarily disable this test and re-enable it after Triton-XPU is upgraded.
@unittest.skip("Temporarily disabled until Triton-XPU upgrade")
class TestDeepSeekOCRTriton(TestDeepSeekOCR):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/DeepSeek-OCR"
        cls.tokenizer = get_tokenizer(cls.model)
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


# Prevent pytest from collecting the imported base test class here.
del TestDeepSeekOCR

if __name__ == "__main__":
    unittest.main()
