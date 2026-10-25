"""DeepSeek-V4.1-Flash PD decode radix cache on the paged KV layout.

Prefill is flushed before every lookup, so `cached_tokens` is the decode-local hit.
"""

import json
import random
import unittest

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-gb300")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
PAGE_SIZE = 256
LAUNCH_TIMEOUT = 1800
NUM_LOOKUPS = 8

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
]
SERVER_ENV = {
    "SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK": "1",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
}
PREFILL_ENV = {
    "SGLANG_ENABLE_PREFILL_WAR_READ_DONE": "1",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "32768",
    "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL": "1",
}
DECODE_ENV = {"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "16384"}


class TestDisaggregationDecodeRadixCacheDSV41(PDDisaggregationServerBase):
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
        cls.tokenizer = get_tokenizer(cls.model)
        cls.extra_prefill_env = {**SERVER_ENV, **PREFILL_ENV}
        cls.extra_decode_env = {**SERVER_ENV, **DECODE_ENV}
        cls.launch_all()

    @classmethod
    def wait_server_ready(cls, url, timeout=LAUNCH_TIMEOUT, process=None):
        super().wait_server_ready(
            url, timeout=max(timeout, LAUNCH_TIMEOUT), process=process
        )

    def _generate(self, input_ids):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
                "routed_dp_rank": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_table(self, rng, num_rows):
        rows = {
            f"key-{i:04d}": str(rng.randrange(10**5, 10**6)) for i in range(num_rows)
        }
        text = "".join(f"{key}: {value}\n" for key, value in rows.items())
        return rows, self._encode(f"Lookup table:\n{text}")

    def test_decode_local_prefix_hit_keeps_lookups_correct(self):
        rng = random.Random(0)
        num_correct = num_total = 0
        for num_rows in (330, 340, 350):
            rows, table_ids = self._build_table(rng, num_rows)
            keys = rng.sample(sorted(rows), NUM_LOOKUPS + 1)
            min_hit = len(table_ids) // PAGE_SIZE * PAGE_SIZE - 2 * PAGE_SIZE
            for lookup_idx, key in enumerate(keys):
                requests.post(
                    f"{self.prefill_url}/flush_cache?timeout=60", timeout=90
                ).raise_for_status()
                output = self._generate(table_ids + self._encode(f"\n{key}:"))
                if lookup_idx > 0:
                    self.assertGreaterEqual(
                        output["meta_info"]["cached_tokens"], min_hit
                    )
                num_correct += output["text"].split()[0] == rows[key]
                num_total += 1
        self.assertGreaterEqual(num_correct, 0.9 * num_total)


if __name__ == "__main__":
    unittest.main()
