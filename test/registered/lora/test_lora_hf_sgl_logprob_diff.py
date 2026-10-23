# Copyright 2023-2024 SGLang Team
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
"""SGLang+LoRA must match HuggingFace+PEFT: top-5 prefill and decode logprobs
within LOGPROB_THRESHOLD on every prompt, on the backend the harness defaults
to, with the decode CUDA graph on and the lm_head logprob path forced to
chunk (chunk size 4) so even short prompts take the multi-pass route."""

import multiprocessing as mp
import os
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

register_cuda_ci(
    est_time=55,
    stage="base-b",
    runner_config="1-gpu-small",
)
register_amd_ci(
    est_time=90,
    suite="stage-b-test-1-gpu-small-amd",
)

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
LORA_PATH = "yushengsu/sglang_lora_logprob_diff_without_tuning"
LOGPROB_THRESHOLD = 1e-01
MAX_NEW_TOKENS = 32
LOGPROB_CHUNK_SIZE = 4
TORCH_DTYPE = torch.float16

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Write a long story.",
    "Write a short story.",
    "What are the main components of a computer?",
]


class TestLoRAHFSGLLogprobDifference(CustomTestCase):
    def test_lora_logprob_comparison_chunked(self):
        lora_paths = [LORA_PATH] * len(PROMPTS)

        # patch.dict restores the environment even if the server fails to start.
        chunk_env = {
            "SGLANG_ENABLE_LOGPROB_CHUNK": "true",
            "SGLANG_LOGPROB_CHUNK_SIZE": str(LOGPROB_CHUNK_SIZE),
        }
        with (
            patch.dict(os.environ, chunk_env),
            SRTRunner(
                BASE_MODEL,
                torch_dtype=TORCH_DTYPE,
                model_type="generation",
                tp_size=1,
                lora_paths=[LORA_PATH],
                max_loras_per_batch=1,
                disable_cuda_graph=False,
                disable_radix_cache=True,
                port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
                mem_fraction_static=0.88,
            ) as srt_runner,
        ):
            srt_outputs = srt_runner.forward(
                PROMPTS, max_new_tokens=MAX_NEW_TOKENS, lora_paths=lora_paths
            )
        torch.cuda.empty_cache()

        with HFRunner(
            BASE_MODEL,
            torch_dtype=TORCH_DTYPE,
            model_type="generation",
            patch_model_do_sample_false=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                PROMPTS, max_new_tokens=MAX_NEW_TOKENS, lora_paths=lora_paths
            )

        for i, prompt in enumerate(PROMPTS):
            prefill_diff = (
                torch.tensor(srt_outputs.top_input_logprobs[i])
                - torch.tensor(hf_outputs.top_input_logprobs[i])
            ).abs()
            decode_diff = (
                torch.tensor(srt_outputs.top_output_logprobs[i])
                - torch.tensor(hf_outputs.top_output_logprobs[i])
            ).abs()
            print(
                f"prompt {i} ({prompt!r}): prefill max/mean diff "
                f"{prefill_diff.max():.3e}/{prefill_diff.mean():.3e}, decode "
                f"{decode_diff.max():.3e}/{decode_diff.mean():.3e}, strings "
                f"{'match' if srt_outputs.output_strs[i].strip() == hf_outputs.output_strs[i].strip() else 'differ'}"
            )
            self.assertLess(
                prefill_diff.max().item(),
                LOGPROB_THRESHOLD,
                f"Prefill logprob mismatch for prompt {i}",
            )
            self.assertLess(
                decode_diff.max().item(),
                LOGPROB_THRESHOLD,
                f"Decode logprob mismatch for prompt {i}",
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
