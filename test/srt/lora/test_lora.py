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
import random
import unittest
from typing import List

import torch
from utils import (
    ALL_OTHER_MULTI_LORA_MODELS,
    CI_MULTI_LORA_MODELS,
    TORCH_DTYPES,
    LoRAModelCase,
    ensure_reproducibility,
)

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l, is_in_ci

TEST_MULTIPLE_BATCH_PROMPTS = [
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
    ### Question 2:
    What do you know about llamas?
    ### Answer:
    """,
    """
    ### Instruction:
    Write a poem about the transformers Python library.
    Mention the word "large language models" in that poem.
    ### Response:
    The Transformers are large language models,
    They're used to make predictions on text.
    """,
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


class TestLoRA(CustomTestCase):
    def _create_test_samples(
        self, lora_adapter_paths: List[str], repeated_trials: int = 3
    ):
        random.seed(42)  # Ensure reproducibility

        patterns = [
            [None, lora_adapter_paths[0], lora_adapter_paths[1]],
            [lora_adapter_paths[0], None, lora_adapter_paths[1]],
            [lora_adapter_paths[0], lora_adapter_paths[1], None],
            [None, lora_adapter_paths[1], None],
            [None, None, None],
        ]

        batches = [
            [random.choice(pattern) for _ in range(3)]
            for pattern in patterns
            for _ in range(repeated_trials)
        ]

        return batches

    def _run_lora_multiple_batch_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            for torch_dtype in TORCH_DTYPES:
                max_new_tokens = 32
                backend = "triton"
                base_path = model_case.base
                lora_adapter_paths = [a.name for a in model_case.adaptors]
                assert len(lora_adapter_paths) >= 2

                print(
                    f"\n========== Testing multiple batches on base '{base_path}' with backend={backend}, dtype={torch_dtype} ---"
                )

                # Initialize runners
                srt_runner = SRTRunner(
                    base_path,
                    torch_dtype=torch_dtype,
                    model_type="generation",
                    lora_paths=[lora_adapter_paths[0], lora_adapter_paths[1]],
                    max_loras_per_batch=len(lora_adapter_paths) + 1,
                    lora_backend=backend,
                    sleep_on_idle=True,  # Eliminate non-determinism by forcing all requests to be processed in one batch.
                    attention_backend="torch_native",
                )
                hf_runner = HFRunner(
                    base_path, torch_dtype=torch_dtype, model_type="generation"
                )

                batches = self._create_test_samples(lora_adapter_paths)
                with srt_runner, hf_runner:
                    for i, lora_paths in enumerate(batches, start=1):
                        prompts = [
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS) for _ in range(3)
                        ]
                        print(
                            f"\n--- Running Batch {i} --- prompts: {prompts}, lora_paths: {lora_paths}"
                        )

                        ensure_reproducibility()
                        srt_outputs = srt_runner.batch_forward(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            lora_paths=lora_paths,
                        )

                        ensure_reproducibility()
                        hf_outputs = hf_runner.forward(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            lora_paths=lora_paths,
                        )

                        print("SRT outputs:", [s for s in srt_outputs.output_strs])
                        print("HF outputs:", [s for s in hf_outputs.output_strs])

                        for srt_out, hf_out in zip(
                            srt_outputs.output_strs, hf_outputs.output_strs
                        ):
                            srt_str = srt_out.strip()
                            hf_str = hf_out.strip()
                            rouge_tol = model_case.rouge_l_tolerance
                            rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
                            if rouge_score < rouge_tol:
                                raise AssertionError(
                                    f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                                    f"for base '{base_path}', adaptor '{lora_paths}', backend '{backend}', prompt: '{prompts}...'"
                                )

                        print(f"--- Batch {i} Comparison Passed --- ")

    def test_ci_lora_models(self):
        self._run_lora_multiple_batch_on_model_cases(CI_MULTI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        filtered_models = []
        for model_case in ALL_OTHER_MULTI_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_lora_multiple_batch_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
