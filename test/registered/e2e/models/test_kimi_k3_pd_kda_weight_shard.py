"""Kimi-K3 (first eight layers, real weights) PD with --kda-weight-shard-size on
the prefill worker. Sharding only moves weights, so each prompt's first token
and its logprob must match the replicated-weight prefill exactly.
"""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
    configure_nixl_pd_backend,
)

register_cuda_ci(est_time=900, stage="nightly", runner_config="4-gpu-gb300")

COMMON_ARGS = [
    "--json-model-override-args",
    json.dumps({"text_config": {"num_hidden_layers": 8}}),
    "--attn-dp-size",
    "2",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--decode-attention-backend",
    "cutedsl_mla",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "64",
    "--mamba-full-memory-ratio",
    "0.2",
    "--chunked-prefill-size",
    "8192",
    "--context-length",
    "32768",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.8",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 6}',
]
# K3's auto-enabled all-reduce fusion asserts under attention DP with TP MoE.
SERVER_ENV = {"SGLANG_K3_AR_FUSION": "0"}
PROMPT_LENGTHS = (100, 1000, 4000, 16000)


def generate(base_url, length, max_new_tokens):
    input_ids = [1] + [1000 + (i * 7919 + length) % 150000 for i in range(length - 1)]
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["meta_info"]["output_token_logprobs"]


class TestKimiK3PDKdaWeightShard(PDDisaggregationServerBase):
    model = "moonshotai/Kimi-K3"
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    extra_decode_args = COMMON_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        configure_nixl_pd_backend(cls)

    def first_tokens(self, prefill_args):
        cls = type(self)
        cls.extra_prefill_args = COMMON_ARGS + prefill_args
        cls.launch_all()
        try:
            first = [generate(self.lb_url, n, 1)[0][:2] for n in PROMPT_LENGTHS]
            with ThreadPoolExecutor(8) as pool:
                burst = list(
                    pool.map(lambda n: generate(self.lb_url, n, 32), PROMPT_LENGTHS * 3)
                )
            self.assertEqual([len(o) for o in burst], [32] * len(burst))
            assert_process_healthy(self, "lb", self.process_lb, self.lb_url)
            assert_process_healthy(
                self, "prefill", self.process_prefill, self.prefill_url
            )
            assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
            return first
        finally:
            cls.tearDownClass()
            cls.setUpClass()

    def test_weight_shard_matches_replicated(self):
        replicated = self.first_tokens([])
        sharded = self.first_tokens(["--kda-weight-shard-size", "2"])
        for n, ref, out in zip(PROMPT_LENGTHS, replicated, sharded):
            self.assertEqual(ref, out, f"prompt_tokens={n}")


if __name__ == "__main__":
    unittest.main()
