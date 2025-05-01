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

import torch

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase


class TestLoRA(CustomTestCase):

    def test_lora_multiple_batch(self):
        base_path = "meta-llama/Llama-3.1-8B-Instruct"
        lora_adapter_paths = [
            "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
        ]
        torch_dtype = torch.float16
        max_new_tokens = 32
        backend = "triton"
        prompts = [
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
            "How does photosynthesis work?",
            "Write a short story.",
            "What are the main components of a computer?",
        ]
        batches = [
            (
                [
                    random.choice(prompts),
                    random.choice(prompts),
                    random.choice(prompts),
                ],
                [
                    None,
                    lora_adapter_paths[0],
                    lora_adapter_paths[1],
                ],
            ),
            (
                [
                    random.choice(prompts),
                    random.choice(prompts),
                    random.choice(prompts),
                ],
                [
                    lora_adapter_paths[0],
                    None,
                    lora_adapter_paths[1],
                ],
            ),
            (
                [
                    random.choice(prompts),
                    random.choice(prompts),
                    random.choice(prompts),
                ],
                [lora_adapter_paths[0], lora_adapter_paths[1], None],
            ),
            (
                [
                    random.choice(prompts),
                    random.choice(prompts),
                    random.choice(prompts),
                ],
                [None, lora_adapter_paths[1], None],
            ),
            (
                [
                    random.choice(prompts),
                    random.choice(prompts),
                    random.choice(prompts),
                ],
                [None, None, None],
            ),
        ]

        print(
            f"\n========== Testing multiple batches on base '{base_path}' with backend={backend}, dtype={torch_dtype} ---"
        )

        # Initialize runners
        srt_runner = SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=[lora_adapter_paths[0], lora_adapter_paths[1]],
            max_loras_per_batch=3,
            lora_backend=backend,
            disable_radix_cache=True,
        )
        hf_runner = HFRunner(
            base_path, torch_dtype=torch_dtype, model_type="generation"
        )

        with srt_runner, hf_runner:
            for i, (prompts, lora_paths) in enumerate(batches):
                truncated_prompts = [p[:20] + "..." for p in prompts]
                print(
                    f"\n--- Running Batch {i+1} --- prompts: {truncated_prompts}, lora_paths: {lora_paths}"
                )

                srt_outputs = srt_runner.batch_forward(
                    prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths
                )

                hf_outputs = hf_runner.forward(
                    prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths
                )

                print("SRT outputs:", [s[:50] + "..." for s in srt_outputs.output_strs])
                print("HF outputs:", [s[:50] + "..." for s in hf_outputs.output_strs])

                # Compare SRT and HF outputs
                self.assertEqual(
                    len(srt_outputs.output_strs), len(hf_outputs.output_strs)
                )
                for srt_out, hf_out in zip(
                    srt_outputs.output_strs, hf_outputs.output_strs
                ):
                    srt_str = srt_out.strip(" ")
                    hf_str = hf_out.strip(" ")
                    self.assertEqual(
                        srt_str, hf_str, f"Mismatch! SRT: '{srt_str}', HF: '{hf_str}'"
                    )

                print(f"--- Batch {i+1} Comparison Passed --- ")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
