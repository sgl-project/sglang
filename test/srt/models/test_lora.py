"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
import uuid

import torch
from vllm.config import LoadConfig

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_config import ModelConfig
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner

LORA_SETS = [
    # {
    #     "base": "meta-llama/Llama-2-7b-hf",
    #     "loras": ["RuterNorway/Llama-2-7b-chat-norwegian-LoRa"],
    # }
    {"base": "meta-llama/Llama-2-7b-hf", "loras": ["winddude/wizardLM-LlaMA-LoRA-7B"]}
    # {"base": "meta-llama/Llama-2-7b-hf", "loras": ["yard1/llama-2-7b-sql-lora-test"]}
]
TORCH_DTYPES = [torch.float16]

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


class TestLoRA(unittest.TestCase):

    def load_lora_adapter(self, lora_set, tp_size):
        base_path = lora_set["base"]
        lora_path = lora_set["loras"][0]

        base_config = ModelConfig(base_path)
        lora_config = LoRAConfig(lora_path)

        uid = uuid.uuid4().hex
        lora_adapter = LoRAAdapter(
            uid, lora_config, base_config, LoadConfig(load_format="auto")
        )
        lora_adapter.initialize_weights()
        print(lora_adapter)

    def inference(self, prompts, lora_set, tp_size, torch_dtype, max_new_tokens):
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = []
        i = 0
        for _ in range(len(prompts)):
            batch_lora_paths.append(all_lora_paths[i])
            i = (i + 1) % len(all_lora_paths)

        with HFRunner(
            base_path,
            torch_dtype=torch_dtype,
            is_generation=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation=True,
            lora_paths=all_lora_paths,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        with HFRunner(
            base_path,
            torch_dtype=torch_dtype,
            is_generation=True,
        ) as hf_runner:
            hf_no_lora_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation=True,
        ) as srt_runner:
            srt_no_lora_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        for i in range(len(prompts)):
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])
            hf_no_lora_logprobs = torch.Tensor(hf_no_lora_outputs.top_input_logprobs[i])
            srt_no_lora_logprobs = torch.Tensor(
                srt_no_lora_outputs.top_input_logprobs[i]
            )
            print(
                "max_diff between hf_lora and srt_lora",
                torch.max(abs(hf_logprobs - srt_logprobs)),
            )
            print(
                "max_diff between srt_base and srt_lora",
                torch.max(abs(srt_no_lora_logprobs - srt_logprobs)),
            )
            print(
                "max_diff between srt_base and hf_base",
                torch.max(abs(srt_no_lora_logprobs - hf_no_lora_logprobs)),
            )
            print(
                "max_diff between hf_lora and hf_base",
                torch.max(abs(hf_logprobs - hf_no_lora_logprobs)),
            )

        print(f"{hf_outputs.output_strs=}")
        print(f"{srt_outputs.output_strs=}")
        print(f"{hf_no_lora_outputs.output_strs=}")
        print(f"{srt_no_lora_outputs.output_strs=}")
        assert hf_outputs.output_strs == srt_outputs.output_strs
        assert hf_no_lora_outputs.output_strs == srt_no_lora_outputs.output_strs

    def base_inference(self, prompts, lora_set, tp_size, torch_dtype, max_new_tokens):
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = [None] * len(prompts)

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation=True,
        ) as srt_runner:
            srt_no_lora_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation=True,
            lora_paths=all_lora_paths,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        for i in range(len(prompts)):
            srt_no_lora_logprobs = torch.Tensor(
                srt_no_lora_outputs.top_input_logprobs[i]
            )
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])
            print("max_diff", torch.max(abs(srt_no_lora_logprobs - srt_logprobs)))

        print(f"{srt_no_lora_outputs.output_strs=}")
        print(f"{srt_outputs.output_strs=}")

        assert srt_outputs.output_strs == hf_outputs.output_strs
        assert srt_no_lora_outputs.output_strs == hf_no_lora_outputs.output_strs

    def test_all(self):
        for lora_set in LORA_SETS:
            # self.load_lora_adapter(lora_set, 1)
            for torch_dtype in TORCH_DTYPES:
                tp_size = 1
                max_new_tokens = 64
                self.inference(PROMPTS, lora_set, tp_size, torch_dtype, max_new_tokens)
                # self.base_inference(
                #     PROMPTS, lora_set, tp_size, torch_dtype, max_new_tokens
                # )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
