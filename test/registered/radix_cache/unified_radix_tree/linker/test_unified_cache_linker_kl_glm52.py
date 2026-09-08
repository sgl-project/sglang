"""GLM-5.2 UnifiedRadixCache direct-linker load-back KL tests."""

import json
import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

GLM52_MODEL = os.environ.get("SGLANG_LINKER_GLM52_MODEL", "zai-org/GLM-5.2-FP8")
GLM52_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="8-gpu-h200")


class TestGLM52UnifiedCacheLinkerKL(UnifiedRadixTreeTestMixin, CustomTestCase):
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
                env={
                    **cls.mooncake.server_env(),
                    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
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

    @unittest.skip("Linker CI targets Direct load-back KL accuracy")
    def test_gsm8k(self):
        pass

    @unittest.skip("Linker CI targets Direct load-back KL accuracy")
    def test_mmlu(self):
        pass

    def prefill_cache_assert(self, result, prefix_len, label):
        self._record_cache_result(result, prefix_len, label)

    def decode_cache_assert(self, result, history_len, output_len, label):
        self._record_cache_result(result, history_len + output_len, label)

    def _record_cache_result(self, result, expected_cached_tokens, label):
        meta_info = result["meta_info"]
        cached_tokens = int(meta_info["cached_tokens"])
        minimum = max(0, expected_cached_tokens - self.page_size)
        self.assertGreaterEqual(
            cached_tokens,
            minimum,
            f"{label}: expected cached_tokens >= {minimum}, got {cached_tokens}",
        )
        details = meta_info.get("cached_tokens_details") or {}
        remote_tokens = int(details.get("host", 0))
        self._direct_remote_tokens += remote_tokens
        if remote_tokens:
            print(f"{label}: Direct load-back confirmed for {remote_tokens} tokens")

    def _run_linker_kl_case(self, test_case):
        self._direct_remote_tokens = 0
        test_case()
        print(f"Direct load-back total: {self._direct_remote_tokens} tokens")
        self.assertGreater(
            self._direct_remote_tokens,
            0,
            "Expected this KL case to load KV through the Mooncake Direct Linker",
        )

    def test_multiturn_logprobs_match(self):
        self._run_linker_kl_case(super().test_multiturn_logprobs_match)

    def test_multiturn_prefill_cache_hit_branching(self):
        self._run_linker_kl_case(super().test_multiturn_prefill_cache_hit_branching)

    def test_multiturn_decode_cache_hit_branching(self):
        self._run_linker_kl_case(super().test_multiturn_decode_cache_hit_branching)


if __name__ == "__main__":
    unittest.main()
