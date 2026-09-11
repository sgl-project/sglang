"""DSV4-Flash 8-GPU PD-Multiplexing sanity (TP8, no spec decoding).

Mirrors test_dsv4_flash_sanity_tp8.py with --enable-pdmux plus its required
companions (overlap schedule off, chunked prefill off). Decode CUDA graphs
stay on and are captured once per SM group; prefill runs the eager
split-prefill path.
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = os.environ.get(
    "SGLANG_TEST_DSV4_FLASH_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)

DSV4_FLASH_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    # Pure-TP (no deepep A2A) resolves the auto MoE runner to Triton, which
    # cannot execute mxfp4-packed experts; dequant them to FP8 at load. No-op
    # for checkpoints whose experts are already FP8.
    "SGLANG_DSV4_FP4_DEQUANT": "1",
}
if not os.path.isdir(DSV4_FLASH_MODEL_PATH):
    # Local checkpoints let model_config auto-detect the routed-expert layout
    # (mxfp4-packed vs converted FP8) from the safetensors header. For the HF
    # slug the header may not be cached yet, so pin the FP8 layout the
    # sgl-project/DeepSeek-V4-Flash-FP8 repo uses.
    DSV4_FLASH_ENV["SGLANG_DSV4_FP4_EXPERTS"] = "0"


class TestDSV4FlashTP8PDMux(
    BasicDecodeCorrectnessMixin,
    CustomTestCase,
):
    """TP8, PD-Multiplexing enabled, no spec decoding."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DSV4_FLASH_MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--enable-pdmux",
                "--disable-overlap-schedule",
                "--chunked-prefill-size",
                "-1",
                "--max-running-requests",
                "8",
                "--mem-fraction-static",
                "0.85",
            ],
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
