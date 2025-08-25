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
import random
import unittest
from typing import List, Optional

import torch
from utils import EMBEDDING_LORA_MODELS, TORCH_DTYPES, LoRAModelCase

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l

PROMPTS = [
    """
### Instruction:
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
""",
    """
### Instruction:
Tell me about llamas and alpacas
### Response:
Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
### Question 2:
What do you know about llamas?
### Answer:
""",
]

EMBEDDING_ADAPTERS = [
    "yard1/llama-2-7b-sql-lora-test"  # target_modules includes embed_tokens
]


class TestLoRALayer(CustomTestCase):

    def ensure_reproducibility(self):
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)

    def _run_embed_layer_test(
        self,
        model_cases: List[LoRAModelCase],
        prompts: List[str],
        batch_lora_paths: List[Optional[str]],
        max_new_tokens: int,
    ):
        for model_case in model_cases:
            for torch_dtype in TORCH_DTYPES:
                backend = "triton"
                base_path = model_case.base
                lora_paths = [a.name for a in model_case.adaptors]

                # Initialize runners
                srt_runner = SRTRunner(
                    base_path,
                    torch_dtype=torch_dtype,
                    model_type="generation",
                    lora_paths=lora_paths,
                    lora_extra_vocab_size=4,
                    max_loras_per_batch=3,
                    disable_cuda_graph=True,
                    lora_backend=backend,
                    sleep_on_idle=True,  # Eliminate non-determinism by forcing all requests to be processed in one batch.
                    attention_backend="torch_native",
                )
                hf_runner = HFRunner(
                    base_path, torch_dtype=torch_dtype, model_type="generation"
                )
                with srt_runner, hf_runner:
                    self.ensure_reproducibility()
                    srt_outputs = srt_runner.batch_forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=batch_lora_paths,
                    )
                    self.ensure_reproducibility()
                    hf_outputs = hf_runner.forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=batch_lora_paths,
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

                    print(f"--- Comparison Passed --- ")

    def test_lora_srt_integration(self):
        """
        Test LoRA integration with SRT runner.
        """
        max_new_tokens = 256
        batch_lora_paths: List[Optional[str]] = [None]
        i = 0
        for _ in range(len(PROMPTS) - 1):
            batch_lora_paths.append(EMBEDDING_ADAPTERS[i])
            i = (i + 1) % len(EMBEDDING_ADAPTERS)

        self._run_embed_layer_test(
            EMBEDDING_LORA_MODELS, PROMPTS, batch_lora_paths, max_new_tokens
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
