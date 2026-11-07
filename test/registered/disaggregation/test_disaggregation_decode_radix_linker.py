"""PD with decode radix cache and the Mooncake direct linker on Kimi Linear (MLA + KDA).

Decode writes the pages it generates to the store, so after both sides flush
their device caches the next turn's prefill restores the previous output too.
"""

import asyncio
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
from sglang.test.test_utils import is_in_ci, unified_radix_tree_server_env

register_cuda_ci(est_time=480, stage="extra-b", runner_config="4-gpu-h100")

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


class TestDisaggregationDecodeRadixLinker(PDDisaggregationServerBase):
    extra_prefill_args = SERVER_ARGS
    extra_decode_args = [*SERVER_ARGS, "--disaggregation-decode-enable-radix-cache"]
    mooncake_p2p_handshake = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MODEL
        cls.mooncake = MooncakeTestServices(
            protocol="rdma" if is_in_ci() else "tcp",
            p2p_handshake=cls.mooncake_p2p_handshake,
        )
        cls.mooncake.start()
        cls.extra_prefill_env = cls._server_env(range(cls.prefill_tp_size))
        cls.extra_decode_env = cls._server_env(
            range(cls.decode_base_gpu_id, cls.decode_base_gpu_id + cls.decode_tp_size)
        )
        cls.launch_all()

    @classmethod
    def _server_env(cls, gpu_indices) -> dict[str, str]:
        return unified_radix_tree_server_env("python", **cls.mooncake.server_env())

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


class TestDisaggregationDecodeRadixLinkerSharedEngine(
    TestDisaggregationDecodeRadixLinker
):
    """The same flow with each store running on its server's PD transfer engine.

    With the default P2P handshake and the PD IB device named for the store too,
    as RDMA deployments configure it, the store reuses the PD transfer engine,
    so the linker and PD transfer register the same GPU KV memory on it.
    """

    mooncake_p2p_handshake = True
    capture_per_side_logs = True

    @classmethod
    def _server_env(cls, gpu_indices) -> dict[str, str]:
        env = super()._server_env(gpu_indices)
        ib_device_args = cls.rdma_devices_for(gpu_indices)
        if ib_device_args:
            env["MOONCAKE_DEVICE"] = ib_device_args[1]
        return env

    @unittest.skipUnless(is_in_ci(), "the store reuses the PD engine only over RDMA")
    def test_stores_reuse_pd_transfer_engine(self):
        for side, stdout, stderr in (
            ("prefill", self._prefill_stdout_buf, self._prefill_stderr_buf),
            ("decode", self._decode_stdout_buf, self._decode_stderr_buf),
        ):
            self.assertIn(
                "Reuse initialized mooncake transfer engine",
                stdout.getvalue() + stderr.getvalue(),
                f"{side} store did not run on the PD transfer engine",
            )


if __name__ == "__main__":
    unittest.main()
