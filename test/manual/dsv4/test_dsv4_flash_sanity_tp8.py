"""DSV4-Flash 8-GPU server sanity (TP8, no spec decoding)."""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.server_sanity_kit import ServerSanityMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}


class TestDSV4FlashTP8NoSpec(ServerSanityMixin, CustomTestCase):
    """TP8, no spec decoding."""

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
