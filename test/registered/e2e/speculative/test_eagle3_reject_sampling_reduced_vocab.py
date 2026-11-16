"""EAGLE3 rejection sampling with a reduced (hot) draft vocabulary.

The Llama-3.1 EAGLE3 draft emits 32000 logits against a 128256-token target.
Rejection sampling carries each step's q at the draft width, through the decode
CUDA graph and the overlap future map, and scatters it onto the target
vocabulary through the d2t map only at verify. A q buffer sized for the target
vocabulary fails graph capture on its first torch.stack.

SGLANG_ENABLE_ASYNC_ASSERT turns a misaligned scatter (q == 0 on a proposed
token) into a crash instead of an inflated accept length. Greedy requests verify
by argmax and never build q, so test_sampled_batch sends sampled requests, nine
of them so the decode batch pads to the next captured size.
"""

import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import SpecCorrectnessKit
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")


class TestEagle3RejectSamplingReducedVocab(Eagle3Base, SpecCorrectnessKit):
    extra_args = ("--speculative-use-rejection-sampling",)
    env_overrides = ((envs.SGLANG_ENABLE_ASYNC_ASSERT, True),)

    def test_sampled_batch(self):
        prompts = [f"Write a short story about the number {i}." for i in range(9)]
        results = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {"temperature": 1.0, "max_new_tokens": 64},
            },
        ).json()
        total_completion, total_verify = 0, 0
        for r in results:
            self.assertIn("text", r, f"Server error: {r}")
            total_completion += r["meta_info"]["completion_tokens"]
            total_verify += r["meta_info"].get("spec_verify_ct", 0)
        self.assertGreater(total_verify, 0)
        # A q supported only on the hot set lowers acceptance but keeps drafts
        # acceptable.
        self.assertGreater(total_completion / total_verify, 1.0)


if __name__ == "__main__":
    unittest.main()
