"""DeepSeek-V4 Flash UnifiedRadixCache direct-linker load-back KL tests."""

import json
import os
import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.linker_load_failure_kit import LinkerLoadFailureMixin
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
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

register_cuda_ci(est_time=378, stage="extra-b", runner_config="4-gpu-h100")


class TestDeepSeekV4FlashUnifiedCacheLinkerKL(
    UnifiedRadixTreeTestMixin, CustomTestCase
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


class TestDeepSeekV4FlashUnifiedCacheLinkerLoadFailure(
    LinkerLoadFailureMixin, TestDeepSeekV4FlashUnifiedCacheLinkerKL
):
    """The same linker arm, with a fraction of its remote KV reads failing.

    Inherits the launch configuration above rather than restating it, so the
    load-backs this faults are the same load-backs the KL arm measures.
    """

    @classmethod
    def setUpClass(cls):
        # Entered before the launch so the server subprocesses inherit it --
        # the injection is read inside the linker, in each TP worker.
        cls._failure_ctx = envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(
            cls.linker_load_failure_prob
        )
        cls._failure_ctx.__enter__()
        try:
            super().setUpClass()
        except Exception:
            cls._failure_ctx.__exit__(None, None, None)
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls._failure_ctx.__exit__(None, None, None)

    # The inherited KL cases cannot express this arm's outcome. Their helpers
    # post a whole batch of prompts as one /generate call and assert the reply
    # has one result per prompt, so a single aborted member turns the reply
    # into an error object and the batch is lost rather than reported. Aborting
    # is the correct behaviour here, so the accuracy signal comes from gsm8k,
    # where every request stands alone and has a known answer.
    _BATCHED = "Batched KL helpers cannot report a per-request abort"

    @unittest.skip(_BATCHED)
    def test_multiturn_logprobs_match(self):
        pass

    @unittest.skip(_BATCHED)
    def test_multiturn_prefill_cache_hit_branching(self):
        pass

    @unittest.skip(_BATCHED)
    def test_multiturn_decode_cache_hit_branching(self):
        pass


if __name__ == "__main__":
    unittest.main()
