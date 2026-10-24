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
"""Chunked input logprobs under TP=2 must match the unchunked run: dropped or
duplicated rows when stitching lm_head chunks across the vocab all-gather."""

import multiprocessing as mp
import unittest

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_LORA_TEST_PROMPTS,
    run_moe_lora_tp_generation,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=130, stage="extra-a", runner_config="2-gpu-large")

# Chunked vs non-chunked runs differ only in lm_head matmul shape (16-row
# chunks vs one full pass), which shifts bf16 rounding slightly.
CHUNK_LOGPROB_THRESHOLD = 5e-03


class TestMoELoRATP2ChunkedLogprobs(CustomTestCase):
    def test_moe_lora_tp2_chunked_vs_unchunked_logprobs(self):
        """Fixing tp_size isolates the chunk grid as the only variable (a TP1
        vs TP2 comparison is polluted by MoE router near-ties on input
        positions). The tiny chunk size forces multi-chunk stitching and the
        per-chunk vocab all-gather under TP=2."""
        prompts = MOE_LORA_TEST_PROMPTS[:3]
        baseline = run_moe_lora_tp_generation(tp_size=2, prompts=prompts)
        torch.cuda.empty_cache()
        with envs.SGLANG_LOGPROB_CHUNK_SIZE.override(16):
            chunked = run_moe_lora_tp_generation(tp_size=2, prompts=prompts)

        for i in range(len(prompts)):
            self.assertEqual(
                baseline["output_strs"][i].strip(),
                chunked["output_strs"][i].strip(),
            )
            base_in = torch.tensor(baseline["top_input_logprobs"][i])
            chunk_in = torch.tensor(chunked["top_input_logprobs"][i])
            self.assertEqual(base_in.shape, chunk_in.shape)
            max_diff = torch.max(torch.abs(base_in - chunk_in)).item()
            self.assertLessEqual(
                max_diff,
                CHUNK_LOGPROB_THRESHOLD,
                f"Chunked vs non-chunked input logprob diff too large on "
                f"prompt {i}: max_diff={max_diff:.6e}",
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
