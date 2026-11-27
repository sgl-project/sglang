"""PD with decode radix cache and the Mooncake direct linker on Kimi Linear (MLA + KDA).

Decode writes the pages it generates to the store, so after both sides flush
their device caches the next turn's prefill restores the previous output too.
"""

import asyncio
import time
import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families

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
from sglang.test.test_utils import is_in_ci, unified_radix_tree_server_env

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")

MODEL = "yujiepan/kimi-linear-tiny-random"
TRACK_INTERVAL = 128

SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "triton",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
    "--page-size",
    "64",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mamba-track-interval",
    str(TRACK_INTERVAL),
    "--mem-fraction-static",
    "0.6",
    "--enable-cache-report",
    "--enable-unified-cache-external-linker",
]


class _DecodeRadixLinkerBase(PDDisaggregationServerBase):
    extra_prefill_args = SERVER_ARGS
    extra_decode_args = [*SERVER_ARGS, "--disaggregation-decode-enable-radix-cache"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MODEL
        cls.mooncake = MooncakeTestServices(protocol="rdma" if is_in_ci() else "tcp")
        cls.mooncake.start()
        env = unified_radix_tree_server_env("python", **cls.mooncake.server_env())
        cls.extra_prefill_env = env
        cls.extra_decode_env = env
        cls.launch_all()

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


class TestDisaggregationDecodeRadixLinker(_DecodeRadixLinkerBase):
    def test_prefill_restores_decode_output_after_flush(self):
        num_rounds, output_len = 4, 600
        history = self._sample_token_ids(input_len=1500, num_prompts=1)[0]
        suffixes = self._sample_token_ids(input_len=300, num_prompts=num_rounds - 1)

        for round_idx in range(num_rounds):
            output = self._generate(history, output_len)
            if round_idx == 0:
                self.assertEqual(output.cached_tokens, 0)
            else:
                decode_end = len(history) - len(suffixes[round_idx - 1]) - 1
                self.assertGreaterEqual(
                    output.cached_tokens,
                    decode_end // TRACK_INTERVAL * TRACK_INTERVAL,
                )
            history.extend(output.output_ids)
            if round_idx < num_rounds - 1:
                history.extend(suffixes[round_idx])
                time.sleep(2)
                self._flush_device_caches()


class TestDisaggregationDecodeLinkerPrefetch(_DecodeRadixLinkerBase):
    extra_prefill_args = [*SERVER_ARGS, "--enable-metrics"]
    extra_decode_args = [
        *_DecodeRadixLinkerBase.extra_decode_args,
        "--unified-cache-external-linker-config",
        '{"load_mode":"prefetch","decode_prefetch":true,"prefetch_policy":"wait_complete"}',
    ]

    def _prefill_transfer_mb(self):
        response = requests.get(f"{self.prefill_url}/metrics", timeout=10)
        response.raise_for_status()
        return sum(
            sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == "sglang:kv_transfer_total_mb_sum"
        )

    def test_decode_restores_prefix_from_store(self):
        (input_ids,) = self._sample_token_ids(input_len=1537, num_prompts=1)

        before = self._prefill_transfer_mb()
        cold = self._generate(input_ids, output_len=16)
        self._flush_device_caches()
        after_cold = self._prefill_transfer_mb()

        warm = self._generate(input_ids, output_len=16)
        self._flush_device_caches()
        after_warm = self._prefill_transfer_mb()

        cold_mb = after_cold - before
        warm_mb = after_warm - after_cold
        self.assertGreater(warm_mb, 0)
        self.assertLess(warm_mb, cold_mb)
        self.assertEqual(warm.output_ids, cold.output_ids)


if __name__ == "__main__":
    unittest.main()
