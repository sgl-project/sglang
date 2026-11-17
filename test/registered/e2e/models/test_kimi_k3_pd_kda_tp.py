"""Kimi-K3 (first eight layers, real weights) PD with --enable-kda-tp and
--shared-experts-tp-size on the decode worker, under both KDA state staging
modes. KDA TP changes the reduction order, so decode logprobs are compared to
the plain layout with a tolerance (observed mean |diff| about 0.03).
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

register_cuda_ci(est_time=1200, stage="nightly", runner_config="4-gpu-gb300")

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
# --shared-experts-tp-size needs an expert-parallel all-to-all backend.
DECODE_ARGS = COMMON_ARGS + ["--ep-size", "2", "--moe-a2a-backend", "megamoe"]
KDA_TP_ARGS = ["--enable-kda-tp", "--shared-experts-tp-size", "2"]
PROMPT_LENGTHS = (100, 1000, 4000, 16000)
MAX_NEW_TOKENS = 16


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
            "top_logprobs_num": 5,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["meta_info"]


class TestKimiK3PDKdaTP(PDDisaggregationServerBase):
    model = "moonshotai/Kimi-K3"
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    # K3's auto-enabled all-reduce fusion asserts under attention DP with TP MoE.
    extra_prefill_env = {"SGLANG_K3_AR_FUSION": "0"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        configure_nixl_pd_backend(cls)

    def outputs(self, prefill_args, decode_args, late_stage="0"):
        cls = type(self)
        cls.extra_prefill_args = COMMON_ARGS + prefill_args
        cls.extra_decode_args = DECODE_ARGS + decode_args
        cls.extra_decode_env = {"SGLANG_K3_KDA_TP_LATE_STAGE": late_stage}
        cls.launch_all()
        try:
            outputs = [generate(self.lb_url, n, MAX_NEW_TOKENS) for n in PROMPT_LENGTHS]
            with ThreadPoolExecutor(8) as pool:
                burst = list(
                    pool.map(lambda n: generate(self.lb_url, n, 32), PROMPT_LENGTHS * 3)
                )
            self.assertEqual(
                [len(m["output_token_logprobs"]) for m in burst], [32] * len(burst)
            )
            assert_process_healthy(self, "lb", self.process_lb, self.lb_url)
            assert_process_healthy(
                self, "prefill", self.process_prefill, self.prefill_url
            )
            assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
            return outputs
        finally:
            cls.tearDownClass()
            cls.setUpClass()

    def assert_close(self, plain, kda_tp):
        diffs = []
        for n, ref, out in zip(PROMPT_LENGTHS, plain, kda_tp):
            ref_tokens = ref["output_token_logprobs"]
            out_tokens = out["output_token_logprobs"]
            # The prefill worker samples the first token; weight sharding is exact.
            self.assertEqual(ref_tokens[0][:2], out_tokens[0][:2], f"prompt {n}")
            for i in range(1, MAX_NEW_TOKENS):
                ref_logprob, ref_id = ref_tokens[i][:2]
                top = {t[1]: t[0] for t in out["output_top_logprobs"][i]}
                self.assertIn(ref_id, top, f"prompt {n}, token {i}")
                diffs.append(abs(ref_logprob - top[ref_id]))
                # A near-tie flip; the contexts differ from here on.
                if out_tokens[i][1] != ref_id:
                    break
        self.assertLess(sum(diffs) / len(diffs), 0.1, diffs)

    def test_kda_tp_matches_plain_layout(self):
        plain = self.outputs([], [])
        for late_stage in ("0", "1"):
            with self.subTest(late_stage=late_stage):
                kda_tp = self.outputs(
                    ["--kda-weight-shard-size", "2"], KDA_TP_ARGS, late_stage
                )
                self.assert_close(plain, kda_tp)


if __name__ == "__main__":
    unittest.main()
