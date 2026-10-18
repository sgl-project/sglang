"""Agens Volundr on a tiny random checkpoint (2 KDA + 1 BCSA + 1 dense layer, Engram, mHC).

The checkpoint's BCSA window is 64 tokens, so every prompt here reaches the far field.
- reference: teacher-forced top-1 and logprobs against values recorded from the checkpoint's own
  transformers implementation (volundr_tiny_reference.json, next to this file)
- decode vs prefill: logprobs of generated tokens match a teacher-forced prefill of the same tokens
"""

import json
import os
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import test_input_output_logprobs_match_helper
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=420, stage="extra-a", runner_config="1-gpu-large")

TINY_MODEL = "Blockway/Agens-Volundr-tiny-random"
REFERENCE = os.path.join(os.path.dirname(__file__), "volundr_tiny_reference.json")


class VolundrTinyChecks:
    """Checks shared by the Volundr tiny-model server configurations."""

    model = TINY_MODEL
    min_top1_agreement = 0.97
    max_mean_logprob_diff = 0.1
    kl_div_thres = 0.002

    def test_reference_logprobs(self):
        with open(REFERENCE) as f:
            ref = json.load(f)
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": ref["input_ids"],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 1,
            },
        ).json()
        meta = resp["meta_info"]
        # Position i predicts token i + 1; the first entry has no logprob.
        top1 = [p[0][1] for p in meta["input_top_logprobs"][1:]]
        target_lp = [x[0] for x in meta["input_token_logprobs"][1:]]
        n = len(ref["top1"])
        agree = sum(a == b for a, b in zip(top1[:n], ref["top1"])) / n
        diff = sum(abs(a - b) for a, b in zip(target_lp[:n], ref["target_logprob"])) / n
        print(f"top-1 agreement {agree:.4f}, mean |dlogprob| {diff:.5f}")
        self.assertGreaterEqual(agree, self.min_top1_agreement)
        self.assertLessEqual(diff, self.max_mean_logprob_diff)

    def test_decode_matches_prefill(self):
        test_input_output_logprobs_match_helper(
            base_url=self.base_url,
            ACC_THRESHOLDS={self.model: {"kl_div": self.kl_div_thres}},
            model_name=self.model,
            max_samples=8,
            max_new_tokens=256,
        )


class TestVolundrTinyFA3(VolundrTinyChecks, DefaultServerBase):
    other_args = ["--attention-backend", "fa3", "--mem-fraction-static", "0.4"]


if __name__ == "__main__":
    unittest.main()
