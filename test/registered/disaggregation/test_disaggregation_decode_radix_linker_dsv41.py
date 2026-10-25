"""DeepSeek-V4.1-Flash PD with decode radix cache and the Mooncake direct linker.

Decode writes the pages it generates to the store, so after both sides flush
their device caches the next turn's prefill restores the previous output too.
"""

import asyncio
import json
import time
import unittest

import requests

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import (
    async_request_sglang_generate,
    gen_payload,
)
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import is_in_ci, try_cached_model

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-gb300")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
PAGE_SIZE = 256
LAUNCH_TIMEOUT = 1800

# The cookbook GB300 disagg recipe (prefill DEP2 / decode DEP16), scaled to 1P1D.
SERVER_ARGS = [
    "--dp-size",
    "2",
    "--ep-size",
    "2",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "megamoe",
    "--dsv4-attn-backend",
    "flashmla",
    "--mem-fraction-static",
    "0.85",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--model-loader-extra-config",
    json.dumps({"enable_multithread_load": True}),
    "--weight-loader-prefetch-checkpoints",
    "--enable-cache-report",
    "--enable-unified-cache-external-linker",
    "--hicache-storage-backend-extra-config",
    json.dumps({"enable_group_semantics": True}),
]
SERVER_ENV = {
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
}
PREFILL_ENV = {
    "SGLANG_ENABLE_PREFILL_WAR_READ_DONE": "1",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "32768",
    "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL": "1",
}
DECODE_ENV = {"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "16384"}


class TestDisaggregationDecodeRadixLinkerDSV41(PDDisaggregationServerBase):
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    extra_prefill_args = [*SERVER_ARGS, "--chunked-prefill-size", "65536"]
    extra_decode_args = [
        *SERVER_ARGS,
        "--disaggregation-decode-enable-radix-cache",
        "--cuda-graph-max-bs-decode",
        "24",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(MODEL)
        cls.mooncake = MooncakeTestServices(
            protocol="rdma" if is_in_ci() else "tcp",
            store_segment_size=32 * 1024**3,
        )
        cls.mooncake.start()
        env = {**SERVER_ENV, **cls.mooncake.server_env()}
        cls.extra_prefill_env = {**env, **PREFILL_ENV}
        cls.extra_decode_env = {**env, **DECODE_ENV}
        cls.launch_all()

    @classmethod
    def wait_server_ready(cls, url, timeout=LAUNCH_TIMEOUT, process=None):
        super().wait_server_ready(
            url, timeout=max(timeout, LAUNCH_TIMEOUT), process=process
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.mooncake.stop()

    def _flush_device_caches(self):
        for url in (self.prefill_url, self.decode_url):
            requests.post(
                f"{url}/flush_cache?timeout=60", timeout=90
            ).raise_for_status()

    def _generate(self, input_ids, output_len):
        output = asyncio.run(
            async_request_sglang_generate(
                gen_payload(input_ids, output_len), f"{self.base_url}/generate"
            )
        )
        self.assertTrue(output.success, output.error)
        return output

    def _sample_token_ids(self, input_len, num_prompts):
        return [
            list(request.prompt)
            for request in sample_random_requests(
                input_len=input_len,
                output_len=1,
                num_prompts=num_prompts,
                range_ratio=1.0,
                tokenizer=get_tokenizer(self.model),
                dataset_path="",
                return_text=False,
            )
        ]

    def test_prefill_restores_decode_output_after_flush(self):
        num_rounds, output_len = 4, 600
        history = self._sample_token_ids(input_len=1500, num_prompts=1)[0]
        # Each follow-up outlasts the 256-token window bounded replay re-prefills.
        suffixes = self._sample_token_ids(input_len=300, num_prompts=num_rounds - 1)

        for round_idx in range(num_rounds):
            output = self._generate(history, output_len)
            if round_idx == 0:
                self.assertEqual(output.cached_tokens, 0)
            else:
                # Decode owns every committed token but the last sampled one.
                decode_end = len(history) - len(suffixes[round_idx - 1]) - 1
                self.assertGreaterEqual(
                    output.cached_tokens, decode_end // PAGE_SIZE * PAGE_SIZE
                )
            history.extend(output.output_ids)
            if round_idx < num_rounds - 1:
                history.extend(suffixes[round_idx])
                time.sleep(2)
                self._flush_device_caches()


if __name__ == "__main__":
    unittest.main()
