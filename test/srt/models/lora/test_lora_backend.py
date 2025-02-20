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

CI_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            ),
        ],
        max_loras_per_batch=1,
    ),
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                prefill_tolerance=1e-1,
            ),
        ],
        max_loras_per_batch=1,
    ),
]

ALL_OTHER_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-2-7b-hf",
        adaptors=[LoRAAdaptor(name="winddude/wizardLM-LlaMA-LoRA-7B")],
        max_loras_per_batch=1,
    ),
]

PROMPTS = [
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
]


class TestLoRABackend(unittest.TestCase):
    def run_backend(
        self,
        prompt: str,
        model_case: LoRAModelCase,
        torch_dtype: torch.dtype,
        max_new_tokens: int,
        backend: str,
    ):
        """
        Run backend tests for a single prompt and model case.
        """
        base_path = model_case.base
        adaptor = model_case.adaptors[0]
        print(
            f"\n========== Testing backend '{backend}' for base '{base_path}' --- "
            f"Prompt '{prompt[:50]}...' using adaptor '{adaptor.name}' ---"
        )
        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=model_case.tp_size,
            lora_paths=[adaptor.name for adaptor in model_case.adaptors],
            max_loras_per_batch=model_case.max_loras_per_batch,
            lora_backend=backend,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            mem_fraction_static=0.88,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                [prompt], max_new_tokens=max_new_tokens, lora_paths=[adaptor.name]
            )

        with HFRunner(
            base_path, torch_dtype=torch_dtype, model_type="generation"
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                [prompt], max_new_tokens=max_new_tokens, lora_paths=[adaptor.name]
            )

        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=model_case.tp_size,
            mem_fraction_static=0.88,
        ) as srt_runner:
            srt_no_lora_outputs = srt_runner.forward(
                [prompt], max_new_tokens=max_new_tokens
            )

        with HFRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
        ) as hf_runner:
            hf_no_lora_outputs = hf_runner.forward(
                [prompt], max_new_tokens=max_new_tokens
            )

        # Use individual adapter tolerances if set, otherwise use model defaults
        prefill_tol = (
            adaptor.prefill_tolerance
            if adaptor.prefill_tolerance is not None
            else model_case.prefill_tolerance
        )
        decode_tol = (
            adaptor.decode_tolerance
            if adaptor.decode_tolerance is not None
            else model_case.decode_tolerance
        )
        rouge_tol = (
            adaptor.rouge_l_tolerance
            if adaptor.rouge_l_tolerance is not None
            else model_case.rouge_l_tolerance
        )

        # Compare prefill stage logprobs (HF vs SRTRunner with LoRA)
        hf_prefill = torch.tensor(hf_outputs.top_input_logprobs[0])
        srt_prefill = torch.tensor(srt_outputs.top_input_logprobs[0])
        max_prefill_diff = torch.max(torch.abs(hf_prefill - srt_prefill))
        print("Max prefill diff (HF vs SRT):", max_prefill_diff)

        # Compare decode stage logprobs
        hf_decode = torch.tensor(hf_outputs.top_output_logprobs[0])
        srt_decode = torch.tensor(srt_outputs.top_output_logprobs[0])
        max_decode_diff = torch.max(torch.abs(hf_decode - srt_decode))
        print("Max decode diff (HF vs SRT):", max_decode_diff)

        srt_output_str = srt_outputs.output_strs[0].strip()
        hf_output_str = hf_outputs.output_strs[0].strip()
        rouge_score = calculate_rouge_l([srt_output_str], [hf_output_str])[0]
        print("ROUGE-L score:", rouge_score)
        print("SRT output:", srt_output_str)
        print("HF output:", hf_output_str)

        # Additional: compare prefill outputs between base model (no LoRA) and LoRA model for reference
        hf_no_lora_prefill = torch.tensor(hf_no_lora_outputs.top_input_logprobs[0])
        srt_no_lora_prefill = torch.tensor(srt_no_lora_outputs.top_input_logprobs[0])
        print(
            "Max diff (SRT base vs SRT LoRA prefill):",
            torch.max(torch.abs(srt_no_lora_prefill - srt_prefill)),
        )
        print(
            "Max diff (HF base vs HF LoRA prefill):",
            torch.max(torch.abs(hf_no_lora_prefill - hf_prefill)),
        )

        if hf_prefill.shape[0] <= 100:
            assert torch.all(torch.abs(hf_prefill - srt_prefill) < prefill_tol), (
                f"Prefill logprobs mismatch for base '{base_path}', adaptor '{adaptor.name}', "
                f"backend '{backend}', prompt: '{prompt[:50]}...'"
            )

        if hf_decode.shape[0] <= 100:
            assert torch.all(torch.abs(hf_decode - srt_decode) < decode_tol), (
                f"Decode logprobs mismatch for base '{base_path}', adaptor '{adaptor.name}', "
                f"backend '{backend}', prompt: '{prompt[:50]}...'"
            )

        if rouge_score < rouge_tol:

            raise AssertionError(
                f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                f"for base '{base_path}', adaptor '{adaptor.name}', backend '{backend}', prompt: '{prompt[:50]}...'"
            )

    def run_backend_batch(
        self,
        prompts: List[str],
        model_case: LoRAModelCase,
        torch_dtype: torch.dtype,
        max_new_tokens: int,
        backend: str,
    ):
        # TODO: Implement batch processing version of run_backend
        raise NotImplementedError(
            "Batch processing version of run_backend is not implemented yet."
        )

    def _run_backend_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in PROMPTS if len(p) < 1000]
            )
            for torch_dtype in TORCH_DTYPES:
                for backend in BACKENDS:
                    for prompt in prompts:
                        self.run_backend(
                            prompt,
                            model_case,
                            torch_dtype,
                            max_new_tokens=32,
                            backend=backend,
                        )

    def test_ci_lora_models(self):
        self._run_backend_on_model_cases(CI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_backend_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
