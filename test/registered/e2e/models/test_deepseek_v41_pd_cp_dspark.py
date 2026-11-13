"""V4.1 prefill CP + PD disaggregation + DSpark on eight B300 GPUs.

GPU 0-3: TP4/CP4 prefill. GPU 4-7: TP4 decode. Both workers use the
DeepSeek-V4.1-Flash checkpoint's bundled DSpark draft and Mooncake transfer.
"""

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

# Initial estimate; update after the first B300 CI run.
register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-b300")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

COMMON_ARGS = [
    "--language-model-only",
    "--attention-backend",
    "dsv4",
    "--moe-a2a-backend",
    "none",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--ep-size",
    "4",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "32",
    "--cuda-graph-max-bs-decode",
    "32",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--context-length",
    "16384",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 6}',
    "--random-seed",
    "0",
]


class TestDeepseekV41PDCPDSpark(PDDisaggregationServerBase, GSM8KMixin):
    model = MODEL
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4

    gsm8k_score_threshold = 0.85
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_num_shots = 8

    extra_prefill_args = COMMON_ARGS + [
        "--enable-prefill-cp",
        "--attn-cp-size",
        "4",
        "--cp-strategy",
        "interleave",
        # Small chunks exercise continuation as well as the P-to-D handoff.
        "--chunked-prefill-size",
        "1024",
        "--max-prefill-tokens",
        "4096",
    ]
    extra_decode_args = COMMON_ARGS
    extra_prefill_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}
    extra_decode_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(MODEL)
        # V4.1's target/draft KV transfer requires Mooncake on both sides.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()

    def _server_info(self, url):
        response = requests.get(url + "/server_info", timeout=30)
        response.raise_for_status()
        return response.json()

    def test_gsm8k(self):
        # Inspect workers directly: the router does not expose both configs.
        for url, role, cp_size in (
            (self.prefill_url, "prefill", 4),
            (self.decode_url, "decode", 1),
        ):
            info = self._server_info(url)
            self.assertEqual(info["disaggregation_mode"], role)
            self.assertEqual(info["disaggregation_transfer_backend"], "mooncake")
            self.assertEqual(info["tp_size"], 4)
            self.assertEqual(info["attn_cp_size"], cp_size)
            self.assertEqual(info["enable_prefill_cp"], role == "prefill")
            self.assertEqual(info["speculative_algorithm"], "DSPARK")
            self.assertEqual(info["speculative_num_draft_tokens"], 6)
            self.assertTrue(info["language_model_only"])

        # Completion evaluation avoids V4.1 chat-template thinking mode.
        # base_url is the PD router, so every request exercises both workers.
        super().test_gsm8k()

        # Require evidence that DSpark actually accepted draft tokens. Query D
        # directly rather than letting the optional mixin check skip a missing
        # metric from the router or the prefill worker.
        states = self._server_info(self.decode_url)["internal_states"]
        self.assertTrue(states, "Decode worker returned no scheduler state")
        for state in states:
            self.assertIn("avg_spec_accept_length", state)
            accept_length = state["avg_spec_accept_length"]
            print(f"V4.1 PD+CP DSpark accept length: {accept_length:.4f}")
            self.assertGreater(accept_length, 1.0)


if __name__ == "__main__":
    unittest.main()
