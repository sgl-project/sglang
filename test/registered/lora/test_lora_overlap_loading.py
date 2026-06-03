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

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    TORCH_DTYPES,
    LoRAAdaptor,
    LoRAModelCase,
    run_lora_test_one_by_one,
)
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

# Two adapters on a freely available base model
LORA_A = "algoprog/fact-generation-llama-3.1-8b-instruct-lora"
LORA_B = "nvidia/llama-3.1-nemoguard-8b-topic-control"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_1 = "AI is a field of computer science focused on"
PROMPT_2 = "The capital of France is"


class TestLoRAOverlapLoadingSingleRequest(CustomTestCase):
    """1. Single request, single LoRA loading with max 1 LoRA in GPU."""

    def test_single_request_single_lora(self):
        model_case = LoRAModelCase(
            base=BASE_MODEL,
            adaptors=[LoRAAdaptor(name=LORA_A)],
            max_loras_per_batch=1,
            max_loaded_loras=1,
        )
        for torch_dtype in TORCH_DTYPES:
            run_lora_test_one_by_one(
                [PROMPT_1],
                model_case,
                torch_dtype,
                max_new_tokens=32,
                enable_lora_overlap_loading=True,
                disable_cuda_graph=True,
                disable_radix_cache=True,
                test_tag="overlap_single_request_single_lora",
            )


class TestLoRAOverlapLoadingBatchReplace(CustomTestCase):
    """2. Two new LoRAs replace two in GPU, batch runs with correct output."""

    def test_two_loras_batch_replace(self):
        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[LORA_A, LORA_B],
            enable_lora_overlap_loading=True,
            max_loras_per_batch=2,
            max_loaded_loras=2,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            sleep_on_idle=True,
        ) as srt_runner:
            # Batch with both LoRAs
            outputs = srt_runner.batch_forward(
                [PROMPT_1, PROMPT_2],
                max_new_tokens=32,
                lora_paths=[LORA_A, LORA_B],
            )
            # Different adapters should produce different outputs
            self.assertNotEqual(
                outputs.output_strs[0].strip(),
                outputs.output_strs[1].strip(),
                "Two different LoRA adapters produced identical output",
            )
            # Both should produce non-empty output
            self.assertTrue(len(outputs.output_strs[0].strip()) > 0)
            self.assertTrue(len(outputs.output_strs[1].strip()) > 0)


class TestLoRAOverlapLoadingEviction(CustomTestCase):
    """3. Two requests: one LoRA already in GPU, one needs loading (eviction).
    Max 2 LoRAs means one must be replaced. Verify correct output."""

    def test_eviction_on_new_lora(self):
        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[LORA_A, LORA_B],
            enable_lora_overlap_loading=True,
            max_loras_per_batch=2,
            max_loaded_loras=2,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            sleep_on_idle=True,
        ) as srt_runner:
            # First batch: load LORA_A into GPU
            out1 = srt_runner.batch_forward(
                [PROMPT_1],
                max_new_tokens=32,
                lora_paths=[LORA_A],
            )
            self.assertTrue(len(out1.output_strs[0].strip()) > 0)

            # Second batch: LORA_A already loaded, LORA_B needs loading
            out2 = srt_runner.batch_forward(
                [PROMPT_1, PROMPT_2],
                max_new_tokens=32,
                lora_paths=[LORA_A, LORA_B],
            )
            # Both should produce non-empty, different outputs
            self.assertTrue(len(out2.output_strs[0].strip()) > 0)
            self.assertTrue(len(out2.output_strs[1].strip()) > 0)
            self.assertNotEqual(
                out2.output_strs[0].strip(),
                out2.output_strs[1].strip(),
                "Different adapters should produce different outputs",
            )

            # Verify LORA_A output is consistent across batches
            self.assertEqual(
                out1.output_strs[0].strip(),
                out2.output_strs[0].strip(),
                "Same adapter + prompt should produce consistent output",
            )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
