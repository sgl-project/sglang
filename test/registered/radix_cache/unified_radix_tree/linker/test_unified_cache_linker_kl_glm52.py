"""GLM-5.2 UnifiedRadixCache direct-linker load-back KL tests."""

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
    unified_radix_tree_server_env,
)

GLM52_MODEL = os.environ.get("SGLANG_LINKER_GLM52_MODEL", "zai-org/GLM-5.2-FP8")
GLM52_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=383, stage="extra-b", runner_config="8-gpu-h200")


class TestGLM52UnifiedCacheLinkerKL(UnifiedCacheLinkerKLTestMixin, CustomTestCase):
    tree_core_backend = "rust"
    page_size = 64
    kl_threshold = 0.03
    sampling_temperature = 0
    max_new_tokens = 64
    prefix_len = 2048
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5

    @classmethod
    def setUpClass(cls):
        cls.model = GLM52_MODEL
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.mooncake.start()
        cls.process = None
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=GLM52_LAUNCH_TIMEOUT,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    "8",
                    "--page-size",
                    str(cls.page_size),
                    "--mem-fraction-static",
                    "0.8",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true, "num_threads": 64}',
                    "--max-total-tokens",
                    "12000",
                    "--max-running-requests",
                    "1",
                    "--enable-cache-report",
                    "--enable-unified-cache-external-linker",
                    "--hicache-storage-backend-extra-config",
                    json.dumps({"enable_group_semantics": True}),
                ],
                env=unified_radix_tree_server_env(
                    cls.tree_core_backend,
                    **cls.mooncake.server_env(),
                ),
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
