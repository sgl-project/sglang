"""SM120 (RTX PRO Blackwell) x DeepSeek-V4-Flash.

Covers the cookbook's "FlashInfer MegaMoE on SM120 for prefill" recipe, run as an
aggregated server (the PD prefill flags are left out). Needs four SM120 GPUs and a
FlashInfer build with the sm120_mxfp4_mxfp8_bf16_cake backend.

DSV4_FLASH_MODEL_PATH overrides the checkpoint (e.g. a local copy).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DSV4FlashAime25TestBase

MODEL = os.environ.get("DSV4_FLASH_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash-0731")


class TestSM120FlashFlashInferMegaMoE(DSV4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--attn-dp-size",
        "4",
        "--moe-a2a-backend",
        "flashinfer_megamoe",
        "--chunked-prefill-size",
        "16384",
        "--max-prefill-tokens",
        "4096",
        "--mem-fraction-static",
        "0.8",
    ]
    EXTRA_ENV = {}

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            raise unittest.SkipTest("needs four CUDA GPUs")
        if torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("needs SM120 (compute capability 12.0)")
        super().setUpClass()


if __name__ == "__main__":
    unittest.main()
