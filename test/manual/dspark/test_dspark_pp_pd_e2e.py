"""Manual PP-prefill/PP1-decode DSpark draft-KV handoff test.

Required:
  SGLANG_TEST_KIMI_K3_MODEL=/path/to/Kimi-K3
  SGLANG_TEST_KIMI_K3_DSPARK_MODEL=/path/to/Kimi-K3-DSpark

Optional topology variables are defined below. The default uses PP2/TP1 prefill
and PP1/TP1 decode on three local GPUs.
"""

import os
import unittest

import requests

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)

TARGET_MODEL = os.environ.get("SGLANG_TEST_KIMI_K3_MODEL")
DRAFT_MODEL = os.environ.get("SGLANG_TEST_KIMI_K3_DSPARK_MODEL")
PREFILL_TP_SIZE = int(os.environ.get("SGLANG_TEST_PREFILL_TP_SIZE", "1"))
PREFILL_PP_SIZE = int(os.environ.get("SGLANG_TEST_PREFILL_PP_SIZE", "2"))
DECODE_TP_SIZE = int(os.environ.get("SGLANG_TEST_DECODE_TP_SIZE", "1"))
DECODE_BASE_GPU_ID = int(
    os.environ.get(
        "SGLANG_TEST_DECODE_BASE_GPU_ID",
        str(PREFILL_TP_SIZE * PREFILL_PP_SIZE),
    )
)
SPEC_BLOCK_SIZE = os.environ.get("SGLANG_TEST_DSPARK_BLOCK_SIZE", "7")

SPEC_ARGS = [
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DRAFT_MODEL or "",
    "--speculative-dspark-block-size",
    SPEC_BLOCK_SIZE,
    "--max-running-requests",
    "8",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


@unittest.skipUnless(
    TARGET_MODEL and DRAFT_MODEL,
    "Set SGLANG_TEST_KIMI_K3_MODEL and SGLANG_TEST_KIMI_K3_DSPARK_MODEL.",
)
class TestDSparkPPPDHandoff(PDDisaggregationServerBase):
    prefill_tp_size = PREFILL_TP_SIZE
    decode_tp_size = DECODE_TP_SIZE
    decode_base_gpu_id = DECODE_BASE_GPU_ID
    extra_prefill_args = [
        "--pp-size",
        str(PREFILL_PP_SIZE),
        "--disable-overlap-schedule",
        *SPEC_ARGS,
    ]
    extra_decode_args = SPEC_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = TARGET_MODEL
        cls.launch_all()

    def test_pp_prefill_to_pp1_decode_draft_kv_handoff(self):
        accept_lengths = []
        for prompt in (
            "Explain why pipeline parallelism needs ordered consensus.",
            "Write a Python function that returns the sum of a list.",
            "Describe the difference between prefill and decode inference.",
        ):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                        "ignore_eos": True,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            meta_info = response.json()["meta_info"]
            self.assertGreater(meta_info["spec_verify_ct"], 0)
            accept_lengths.append(meta_info["spec_accept_length"])

        self.assertGreater(max(accept_lengths), 1.0)
        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
