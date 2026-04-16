"""
python3 -m unittest test_deepseek_ocr_triton.py
"""

import os
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# Import base test class
from test_deepseek_ocr import TestDeepSeekOCR

# Register for per-commit XPU tests (higher est_time to run before test_deepseek_ocr.py)
register_xpu_ci(est_time=400, suite="per-commit-xpu")
# Register for nightly XPU tests
register_xpu_ci(est_time=400, suite="nightly-xpu", nightly=True)


class TestDeepSeekOCRTriton(TestDeepSeekOCR):
    @classmethod
    def setUpClass(cls):
        cls._cleanup_xpu_memory()
        cls.model = "deepseek-ai/DeepSeek-OCR"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, use_fast=False, trust_remote_code=True
        )
        cls.base_url = "http://127.0.0.1:21002"
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
            "--mem-fraction-static",
            "0.7",
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

    @classmethod
    def tearDownClass(cls):
        """Ensure server process is killed after tests."""
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        cls._cleanup_xpu_memory()


if __name__ == "__main__":
    unittest.main()
