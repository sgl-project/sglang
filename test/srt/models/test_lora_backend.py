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

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import calculate_rouge_l

LORA_SETS = [
    {"base": "meta-llama/Llama-2-7b-hf", "loras": ["winddude/wizardLM-LlaMA-LoRA-7B"]},
    {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "loras": ["reissbaker/llama-3.1-8b-abliterated-lora"],
        "decode_tolerance": 8e-2,
    },
]
TORCH_DTYPES = [torch.float16]

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

BACKENDS = ["triton", "flashinfer"]

prefill_tolerance: float = 5e-2
decode_tolerance: float = 5e-2
rouge_l_tolerance: float = 1


class TestLoRABackend(unittest.TestCase):

    def run_backend(
        self, prompts, lora_set, tp_size, torch_dtype, max_new_tokens, backend
    ):
        print(f"=================== testing {backend} backend =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = []
        i = 0
        for _ in range(len(prompts)):
            batch_lora_paths.append(all_lora_paths[i])
            i = (i + 1) % len(all_lora_paths)
        print(f"batch lora paths={batch_lora_paths}")
        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=tp_size,
            lora_paths=all_lora_paths,
            max_loras_per_batch=3,
            lora_backend=backend,
            disable_cuda_graph=True,
            disable_radix_cache=True,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        with HFRunner(
            base_path, torch_dtype=torch_dtype, model_type="generation"
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
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

        for i in range(len(prompts)):
            print(f"Prompt {i} with lora path {batch_lora_paths[i]}:")

            # compare input logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])
            hf_no_lora_logprobs = torch.Tensor(hf_no_lora_outputs.top_input_logprobs[i])
            srt_no_lora_logprobs = torch.Tensor(
                srt_no_lora_outputs.top_input_logprobs[i]
            )
            print(
                "max input diff between hf_lora and srt_lora",
                torch.max(abs(hf_logprobs - srt_logprobs)),
            )
            print(
                "max input diff between srt_base and srt_lora",
                torch.max(abs(srt_no_lora_logprobs - srt_logprobs)),
            )
            print(
                "max input diff between srt_base and hf_base",
                torch.max(abs(srt_no_lora_logprobs - hf_no_lora_logprobs)),
            )
            print(
                "max input diff between hf_lora and hf_base",
                torch.max(abs(hf_logprobs - hf_no_lora_logprobs)),
            )
            if hf_logprobs.shape[0] <= 100:
                tol = lora_set.get("prefill_tolerance", prefill_tolerance)
                assert torch.all(abs(hf_logprobs - srt_logprobs) < tol), (
                    f"prefill logprobs are not all close with model_path={base_path},"
                    f"lora_path={batch_lora_paths[i]}, backend={backend}, prompt={prompts[i]}"
                    f"prefill_tolerance={prefill_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )

            # compare output logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_output_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_output_logprobs[i])
            print(
                "max output diff between hf_lora and srt_lora",
                torch.max(abs(hf_logprobs - srt_logprobs)),
                "\n",
            )
            if hf_logprobs.shape[0] <= 100:
                tol = lora_set.get("decode_tolerance", decode_tolerance)
                assert torch.all(abs(hf_logprobs - srt_logprobs) < tol), (
                    f"decode logprobs are not all close with model_path={base_path},"
                    f"lora_path={batch_lora_paths[i]}, backend={backend}, prompt={prompts[i]}"
                    f"decode_tolerance={decode_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )

            # compare output strings
            srt_output_str = srt_outputs.output_strs[i].strip(" ")
            hf_output_str = hf_outputs.output_strs[i].strip(" ")
            print(f"srt_output_str={srt_output_str}")
            print(f"hf_output_str={hf_output_str}")
            rouge_l_scores = calculate_rouge_l([srt_output_str], [hf_output_str])
            print(f"{rouge_l_scores=}")
            assert (
                rouge_l_scores[0] >= rouge_l_tolerance
            ), f"ROUGE-L scores of prompt {i} outputs are greater than rouge_l_tolerance={rouge_l_tolerance}"

    def test_all(self):
        for lora_set in LORA_SETS:
            print(f"Testing lora set {lora_set}: ")
            for torch_dtype in TORCH_DTYPES:
                tp_size = 1
                max_new_tokens = 32
                for backend in BACKENDS:
                    self.run_backend(
                        PROMPTS, lora_set, tp_size, torch_dtype, max_new_tokens, backend
                    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
