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

from utils import (
    BACKENDS,
    TORCH_DTYPES,
    LoRAAdaptor,
    LoRAModelCase,
    run_batch_lora_test,
)

from sglang.test.test_utils import CustomTestCase, is_in_ci

CI_MULTI_LORA_MODELS = [
    # multi-rank case
    LoRAModelCase(
        base="meta-llama/Llama-2-7b-hf",
        adaptors=[
            LoRAAdaptor(
                name="winddude/wizardLM-LlaMA-LoRA-7B",
                prefill_tolerance=1e-1,
            ),
            LoRAAdaptor(
                name="RuterNorway/Llama-2-7b-chat-norwegian-LoRa",
                prefill_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=2,
    ),
]

ALL_OTHER_MULTI_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                prefill_tolerance=1e-1,
            ),
            LoRAAdaptor(
                name="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                prefill_tolerance=1e-1,
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


class TestMultiLoRABackend(CustomTestCase):

    def _run_multi_lora_test_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters.
            batch_prompts = (
                PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in PROMPTS if len(p) < 1000]
            )
            for torch_dtype in TORCH_DTYPES:
                for backend in BACKENDS:
                    run_batch_lora_test(
                        batch_prompts,
                        model_case,
                        torch_dtype,
                        max_new_tokens=32,
                        backend=backend,
                        test_tag="multi-lora-backend",
                    )

    def test_ci_lora_models(self):
        self._run_multi_lora_test_on_model_cases(CI_MULTI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_MULTI_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_multi_lora_test_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
