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
    ALL_OTHER_LORA_MODELS,
    CI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TORCH_DTYPES,
    LoRAModelCase,
    run_lora_test_by_batch,
    run_lora_test_one_by_one,
)

from sglang.test.test_utils import CustomTestCase, is_in_ci

TEST_CUDA_GRAPH_PADDING_PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
    ### Question 2:
    What do you know about llamas?
    ### Answer:
    """,
    "Computer science is the study of",
]


class TestLoRACudaGraph(CustomTestCase):

    def _run_without_cuda_graph_on_model_cases(self, model_cases: List[LoRAModelCase]):
        # Since we have already enabled CUDA graph by default in other lora tests,
        # we only need to run lora tests without CUDA graph here.
        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                DEFAULT_PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            )
            for torch_dtype in TORCH_DTYPES:
                run_lora_test_one_by_one(
                    prompts,
                    model_case,
                    torch_dtype,
                    max_new_tokens=32,
                    backend="triton",
                    disable_cuda_graph=True,
                    test_tag="without_cuda_graph",
                )

    def _run_cuda_graph_padding_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            # Run a batch size of 3, which will not be captured by CUDA graph and need padding
            prompts = TEST_CUDA_GRAPH_PADDING_PROMPTS
            for torch_dtype in TORCH_DTYPES:
                run_lora_test_by_batch(
                    prompts,
                    model_case,
                    torch_dtype,
                    max_new_tokens=32,
                    backend="triton",
                    disable_cuda_graph=False,
                    test_tag="cuda_graph_padding",
                )

    def test_ci_lora_models(self):
        self._run_without_cuda_graph_on_model_cases(CI_LORA_MODELS)
        self._run_cuda_graph_padding_on_model_cases(CI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_without_cuda_graph_on_model_cases(filtered_models)
        self._run_cuda_graph_padding_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
