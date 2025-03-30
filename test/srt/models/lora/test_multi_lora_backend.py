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
from typing import List

import torch
from utils import BACKENDS, TORCH_DTYPES, LoRAAdaptor, LoRAModelCase

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l, is_in_ci

MULTI_LORA_MODELS = [
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
        base_path = model_case.base
        adaptor_names = [adaptor.name for adaptor in model_case.adaptors]
        print(
            f"\n========== Testing multi-LoRA backend '{backend}' for base '{model_case.base}' --- "
            f"Using prompts {[p[:50] for p in prompts]} with adaptors: {adaptor_names} ---"
        )
        print(
            "run_backend_batch: Multi-LoRA backend test functionality is pending support."
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
                prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
            )

        with HFRunner(
            base_path, torch_dtype=torch_dtype, model_type="generation"
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
            )

        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=model_case.tp_size,
            mem_fraction_static=0.88,
        ) as srt_runner:
            srt_no_lora_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        with HFRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
        ) as hf_runner:
            hf_no_lora_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        # Compare prefill stage logprobs (HF vs SRTRunner with LoRA)
        for i in range(len(prompts)):
            adaptor = model_case.adaptors[i]
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
            hf_prefill = torch.tensor(hf_outputs.top_input_logprobs[i])
            srt_prefill = torch.tensor(srt_outputs.top_input_logprobs[i])
            max_prefill_diff = torch.max(torch.abs(hf_prefill - srt_prefill))
            print("Max prefill diff (HF vs SRT):", max_prefill_diff)

            # Compare decode stage logprobs
            hf_decode = torch.tensor(hf_outputs.top_output_logprobs[i])
            srt_decode = torch.tensor(srt_outputs.top_output_logprobs[i])
            max_decode_diff = torch.max(torch.abs(hf_decode - srt_decode))
            print("Max decode diff (HF vs SRT):", max_decode_diff)

            srt_output_str = srt_outputs.output_strs[i].strip()
            hf_output_str = hf_outputs.output_strs[i].strip()
            rouge_score = calculate_rouge_l([srt_output_str], [hf_output_str])[0]
            print("ROUGE-L score:", rouge_score)
            print("SRT output:", srt_output_str)
            print("HF output:", hf_output_str)

            # Additional: compare prefill outputs between base model (no LoRA) and LoRA model for reference
            hf_no_lora_prefill = torch.tensor(hf_no_lora_outputs.top_input_logprobs[i])
            srt_no_lora_prefill = torch.tensor(
                srt_no_lora_outputs.top_input_logprobs[i]
            )
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
                    f"Prefill logprobs mismatch for base '{base_path}', adaptor '{adaptor_names}', "
                    f"backend '{backend}', prompt: '{prompts[0][:50]}...'"
                )

            if hf_decode.shape[0] <= 100:
                assert torch.all(torch.abs(hf_decode - srt_decode) < decode_tol), (
                    f"Decode logprobs mismatch for base '{base_path}', adaptor '{adaptor_names}', "
                    f"backend '{backend}', prompt: '{prompts[0][:50]}...'"
                )

            if rouge_score < rouge_tol:
                raise AssertionError(
                    f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                    f"for base '{base_path}', adaptor '{adaptor_names}', backend '{backend}', prompt: '{prompts[0][:50]}...'"
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
