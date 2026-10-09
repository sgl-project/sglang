"""DeepSeek-V4 Flash UnifiedRadixCache direct-linker load-back KL tests."""

import json
import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedCacheLinkerKLTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

DSV4_FLASH_MODEL = os.environ.get(
    "SGLANG_LINKER_DSV4_FLASH_MODEL", "sgl-project/DeepSeek-V4-Flash-FP8"
)
DSV4_FLASH_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=210, stage="extra-b", runner_config="4-gpu-h100")


class TestDeepSeekV4FlashUnifiedCacheLinkerKL(
    UnifiedCacheLinkerKLTestMixin, CustomTestCase
):
    page_size = 256
    kl_threshold = 0.01
    sampling_temperature = 0
    max_new_tokens = 64
    prefix_len = 2048
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.mooncake.start()
        cls.process = None
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    "4",
                    "--attention-backend",
                    "compressed",
                    "--page-size",
                    str(cls.page_size),
                    "--chunked-prefill-size",
                    "8192",
                    "--mem-fraction-static",
                    "0.92",
                    "--disable-shared-experts-fusion",
                    "--swa-full-tokens-ratio",
                    "0.25",
                    "--max-total-tokens",
                    "8192",
                    "--max-running-requests",
                    "1",
                    "--enable-cache-report",
                    "--enable-unified-cache-external-linker",
                    "--hicache-storage-backend-extra-config",
                    json.dumps({"enable_group_semantics": True}),
                ],
                env={
                    **cls.mooncake.server_env(),
                    "SGLANG_DSV4_FP4_EXPERTS": "0",
                },
            )
            cls.input_ids = get_input_ids(cls.model, num_samples=18)
        except Exception:
            try:
                if cls.process is not None:
                    terminate_and_kill_process_tree(cls.process)
            finally:
                cls.mooncake.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.process is not None:
                terminate_and_kill_process_tree(cls.process)
        finally:
            cls.mooncake.stop()


if __name__ == "__main__":
    unittest.main()
