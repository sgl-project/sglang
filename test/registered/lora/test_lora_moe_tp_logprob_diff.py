# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MoE LoRA under TP=2 must match the vLLM reference: wrong expert-LoRA
sharding or a missing reduce changes tokens or logprobs with no error."""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_LORA_TEST_PROMPTS,
    MOE_LORA_VLLM_REFERENCE,
    MOE_LORA_VLLM_REFERENCE_STATS,
    run_moe_lora_tp_generation,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=65, stage="base-b", runner_config="2-gpu-large")

# The TP=1 run stays within stats + 2e-4 of vLLM and TP=2 within 5e-4 of TP=1.
LOGPROB_EPS = 7e-04


class TestMoELoRATP2Logprobs(CustomTestCase):
    def test_moe_lora_tp2_matches_vllm_reference(self):
        tp2 = run_moe_lora_tp_generation(tp_size=2, prompts=MOE_LORA_TEST_PROMPTS)

        for i, prompt in enumerate(MOE_LORA_TEST_PROMPTS):
            ref = MOE_LORA_VLLM_REFERENCE[i]
            self.assertEqual(
                ref["text"].strip(),
                tp2["output_strs"][i].strip(),
                f"String mismatch on prompt {i} ({prompt[:40]!r})",
            )
            lps = torch.tensor(
                [
                    t[0] if isinstance(t, list) else t
                    for t in tp2["top_output_logprobs"][i]
                ]
            )
            ref_lps = torch.tensor(ref["lps"])
            # One logprob per generated token on both sides, including the EOS
            # token that ends a short answer: compare the whole reference, never
            # a common prefix.
            self.assertGreater(lps.numel(), 0, f"No output logprobs on prompt {i}")
            self.assertEqual(
                lps.shape, ref_lps.shape, f"Logprob count mismatch on prompt {i}"
            )
            self.assertTrue(
                torch.isfinite(lps).all(), f"Non-finite logprob on prompt {i}"
            )
            diff = torch.abs(lps - ref_lps)
            stats = MOE_LORA_VLLM_REFERENCE_STATS[i]
            self.assertLessEqual(
                diff.max().item(),
                stats["max"] + LOGPROB_EPS,
                f"Max logprob diff exceeded on prompt {i}",
            )
            self.assertLessEqual(
                diff.mean().item(),
                stats["mean"] + LOGPROB_EPS,
                f"Mean logprob diff exceeded on prompt {i}",
            )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
