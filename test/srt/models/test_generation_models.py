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
"""
Usage:

To test a specific model locally:
1. Add it to ALL_MODELS, for example, `ModelCase("Qwen/Qwen2-1.5B")`
2. Run `ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels`
"""

import dataclasses
import multiprocessing as mp
import os
import random
import unittest
from typing import List

import torch

from sglang.test.runners import (
    DEFAULT_PROMPTS,
    HFRunner,
    SRTRunner,
    check_close_model_outputs,
)
from sglang.test.test_utils import CustomTestCase, is_in_ci


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 6e-2  # Increased to fix numerical error in issue #8614.
    rouge_l_tolerance: float = 1
    skip_long_prompt: bool = False
    trust_remote_code: bool = False


# Popular models that run on the CI
CI_MODELS = [
    ModelCase("meta-llama/Llama-3.1-8B-Instruct"),
    ModelCase("google/gemma-2-2b"),
]

# the complete set of models to test sglang's generation model
ALL_MODELS = [
    *CI_MODELS,
    ModelCase("Qwen/Qwen2-1.5B"),
    ModelCase("Qwen/Qwen2.5-14B-Instruct"),
    ModelCase("HuggingFaceTB/SmolLM-135M-Instruct", skip_long_prompt=True),
    ModelCase("allenai/OLMo-1B-0724-hf", decode_tolerance=8e-2, skip_long_prompt=True),
    ModelCase("shanearora/2025-sep-a-base-model"),
    ModelCase(
        "THUDM/glm-4-9b-chat", tp_size=2, trust_remote_code=True, skip_long_prompt=True
    ),
    ModelCase("openai-community/gpt2"),
    ModelCase("microsoft/phi-1_5", trust_remote_code=True),
    ModelCase("adept/persimmon-8b-chat"),
    ModelCase("upstage/SOLAR-10.7B-Instruct-v1.0"),
    ModelCase("inclusionAI/Ling-lite", trust_remote_code=True),
    ModelCase("microsoft/Phi-3-small-8k-instruct", trust_remote_code=True),
    ModelCase("allenai/OLMo-2-1124-7B-Instruct", skip_long_prompt=True),
    ModelCase("ibm-granite/granite-3.0-2b-instruct", skip_long_prompt=True),
    ModelCase(
        "microsoft/Phi-3.5-MoE-instruct",
        tp_size=2,
        trust_remote_code=True,
        skip_long_prompt=True,
    ),
    ModelCase("facebook/opt-125m", skip_long_prompt=True),
    ModelCase(
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        tp_size=2,
        trust_remote_code=True,
        skip_long_prompt=True,
    ),
    ModelCase(
        "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        tp_size=8,
        trust_remote_code=True,
        skip_long_prompt=True,
    ),
    ModelCase(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        trust_remote_code=True,
        skip_long_prompt=True,
    ),
    ModelCase(
        "swiss-ai/Apertus-8B",
        trust_remote_code=True,
        skip_long_prompt=True,
    ),
]

TORCH_DTYPES = [torch.float16]


class TestGenerationModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_logits_and_output_strs(
        self,
        prompts: List[str],
        model_case: ModelCase,
        torch_dtype: torch.dtype,
    ) -> None:
        model_path = model_case.model_path
        prefill_tolerance, decode_tolerance, rouge_l_tolerance = (
            model_case.prefill_tolerance,
            model_case.decode_tolerance,
            model_case.rouge_l_tolerance,
        )
        max_new_tokens = 32

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens)

        with SRTRunner(
            model_path,
            tp_size=model_case.tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=model_case.prefill_tolerance,
            decode_tolerance=model_case.decode_tolerance,
            rouge_l_tolerance=model_case.rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={prompts}",
        )

    @unittest.skipIf(not is_in_ci(), "Local test should run all models")
    def test_ci_models(self):
        for model_case in CI_MODELS:
            for torch_dtype in TORCH_DTYPES:
                prompts = DEFAULT_PROMPTS

                # Skip long prompts for models that do not have a long context
                if model_case.skip_long_prompt:
                    prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

                # Assert the logits and output strs are close
                self.assert_close_logits_and_output_strs(
                    prompts, model_case, torch_dtype
                )

    @unittest.skipIf(is_in_ci(), "CI only runs selected models for simplicity")
    def test_all_models(self):
        for model_case in ALL_MODELS:
            for torch_dtype in TORCH_DTYPES:
                if (
                    "ONLY_RUN" in os.environ
                    and os.environ["ONLY_RUN"] != model_case.model_path
                ):
                    continue

                # Skip long prompts for models that do not have a long context
                prompts = DEFAULT_PROMPTS
                if model_case.skip_long_prompt:
                    prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

                # Assert the logits and output strs are close
                self.assert_close_logits_and_output_strs(
                    prompts, model_case, torch_dtype
                )


if __name__ == "__main__":
    unittest.main()
