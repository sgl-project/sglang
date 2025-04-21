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

import dataclasses
from typing import List

import torch

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import calculate_rouge_l


@dataclasses.dataclass
class LoRAAdaptor:
    name: str
    prefill_tolerance: float = None
    decode_tolerance: float = None
    rouge_l_tolerance: float = None


@dataclasses.dataclass
class LoRAModelCase:
    base: str
    adaptors: List[LoRAAdaptor]
    tp_size: int = 1
    prefill_tolerance: float = 1e-1
    decode_tolerance: float = 1e-1
    rouge_l_tolerance: float = 1.0
    max_loras_per_batch: int = 1
    skip_long_prompt: bool = False

    def __post_init__(self):
        if len(self.adaptors) > self.max_loras_per_batch:
            raise ValueError(
                f"For base '{self.base}', number of adaptors ({len(self.adaptors)}) "
                f"must be <= max_loras_per_batch ({self.max_loras_per_batch})"
            )


TORCH_DTYPES = [torch.float16]
BACKENDS = ["triton"]
DEFAULT_PROMPTS = [
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
]

ALL_OTHER_LORA_MODELS = [
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
    LoRAModelCase(
        base="meta-llama/Llama-2-7b-hf",
        adaptors=[LoRAAdaptor(name="winddude/wizardLM-LlaMA-LoRA-7B")],
        max_loras_per_batch=2,
    ),
]


def run_batch_lora_test(
    prompts: List[str],
    model_case: LoRAModelCase,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    backend: str,
    disable_cuda_graph: bool = True,
    disable_radix_cache: bool = True,
    mem_fraction_static: float = 0.88,
    test_tag: str = "",
):
    """
    Run Lora test for a forward batch.
    For prompt0, prompt1, ..., promptN,
    we will use adaptor0, adaptor1, ..., adaptorN included in model case,
    We will then compare the outputs of HF and SRT with and without LoRA.
    If number of prompts is larger than number of adaptors,
    the prompt i will use adaptor i % (number of adaptors).

    Args:
        prompts (List[str]): The batch of prompts to test.
        model_case (LoRAModelCase): The model case to test.
        torch_dtype (torch.dtype): The torch dtype to use.
        max_new_tokens (int): The maximum number of new tokens to generate.
        backend (str): The lora backend to use.
        disable_cuda_graph (bool, optional): Whether to disable CUDA graph. Defaults to True.
        disable_radix_cache (bool, optional): Whether to disable radix cache. Defaults to True.
        mem_fraction_static (float, optional): The fraction of memory to use. Defaults to 0.88.
        test_tag (str, optional): The tag to use for the test. Defaults to "".
    """
    base_path = model_case.base

    # Create used adaptors for each prompt in batch
    i, adaptors = 0, []
    for _ in range(len(prompts)):
        adaptors.append(model_case.adaptors[i])
        i = (i + 1) % len(model_case.adaptors)
    adaptor_names = [adaptor.name for adaptor in adaptors]

    print(
        f"\n========== Testing {test_tag} on base '{model_case.base}' with backend={backend}, dtype={torch_dtype} --- "
        f"Using prompts {[p[:50] for p in prompts]} with adaptors: {adaptor_names} ---"
    )
    with SRTRunner(
        base_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=model_case.tp_size,
        lora_paths=[adaptor.name for adaptor in model_case.adaptors],
        max_loras_per_batch=model_case.max_loras_per_batch,
        lora_backend=backend,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=disable_radix_cache,
        mem_fraction_static=mem_fraction_static,
    ) as srt_runner:
        srt_outputs = srt_runner.forward(
            prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
        )

    with SRTRunner(
        base_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=model_case.tp_size,
        mem_fraction_static=mem_fraction_static,
    ) as srt_runner:
        srt_no_lora_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

    with HFRunner(
        base_path, torch_dtype=torch_dtype, model_type="generation"
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
        )
        hf_no_lora_outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens)

    # Compare prefill stage logprobs (HF vs SRTRunner with LoRA)
    for i in range(len(prompts)):
        adaptor = adaptors[i]
        # Use individual adaptor tolerances if set, otherwise use model defaults
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
        srt_no_lora_prefill = torch.tensor(srt_no_lora_outputs.top_input_logprobs[i])
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
