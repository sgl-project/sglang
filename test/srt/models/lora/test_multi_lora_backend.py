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
import os
import unittest
from typing import List

import torch
from utils import *

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import calculate_rouge_l, is_in_ci

MULTI_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            ),
            LoRAAdaptor(
                name="some-org/another-lora-adaptor",
            ),
        ],
        max_loras_per_batch=2,
    ),
]

# All prompts are used at once in a batch.
PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids.
    ### Question:
    What do you know about llamas?
    ### Answer:
    """,
]


class TestMultiLoRABackend(unittest.TestCase):
    def run_backend_batch(
        self,
        prompts: List[str],
        model_case: LoRAModelCase,
        torch_dtype: torch.dtype,
        max_new_tokens: int,
        backend: str,
    ):
        """
        The multi-LoRA backend test functionality is not supported yet.
        This function uses all prompts at once and prints a message indicating that support is pending.
        """
        adaptor_names = [adaptor.name for adaptor in model_case.adaptors]
        print(
            f"\n========== Testing multi-LoRA backend '{backend}' for base '{model_case.base}' --- "
            f"Using prompts {[p[:50] for p in prompts]} with adaptors: {adaptor_names} ---"
        )
        print(
            "run_backend_batch: Multi-LoRA backend test functionality is pending support."
        )

    def _run_backend_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters.
            batch_prompts = (
                PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in PROMPTS if len(p) < 1000]
            )
            for torch_dtype in TORCH_DTYPES:
                for backend in BACKENDS:
                    self.run_backend_batch(
                        batch_prompts,
                        model_case,
                        torch_dtype,
                        max_new_tokens=32,
                        backend=backend,
                    )

    def test_multi_lora_models(self):
        # Optionally skip tests in CI environments.
        if is_in_ci():
            return
        self._run_backend_on_model_cases(MULTI_LORA_MODELS)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
