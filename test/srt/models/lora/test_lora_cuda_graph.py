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

LORA_SETS = [
    # {
    #     "base": "meta-llama/Llama-2-7b-hf",
    #     "loras": ["RuterNorway/Llama-2-7b-chat-norwegian-LoRa"],
    # },
    # {"base": "meta-llama/Llama-2-7b-hf", "loras": ["winddude/wizardLM-LlaMA-LoRA-7B"]},
    {
        "base": "meta-llama/Llama-2-7b-hf",
        "loras": [
            "LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128",
            "LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens",
        ],
    },
    # {"base": "baichuan-inc/Baichuan-7B", "loras": ["baichuan7B-zero"]},
    # {"base": "mistralai/Mistral-7B-Instruct-v0.3", "loras": ["/home/ying/test_lora"]},
    # {
    #     "base": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "loras": [
    #         "/home/ying/test_lora",
    #         "/home/ying/test_lora_1",
    #         "/home/ying/test_lora_2",
    #         "/home/ying/test_lora_3",
    #         "/home/ying/test_lora_4",
    #     ],
    # },
    # {"base": "meta-llama/Llama-2-7b-hf", "loras": ["yard1/llama-2-7b-sql-lora-test"]},
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
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
""",
]

# import json
#
# with open("/home/ying/test_prompt/dialogue_choice_prompts.json", "r") as f:
#     samples = json.load(f)
# for sample in samples[:5]:
#     assert sample[0]["role"] == "user"
#     PROMPTS.append(sample[0]["content"][:2000])


class TestLoRA(unittest.TestCase):

    def inference(self, prompts, lora_set, tp_size, torch_dtype, max_new_tokens):
        print("=================== testing inference =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = all_lora_paths
        # i = 0
        # for _ in range(len(prompts) - 1):
        #     batch_lora_paths.append(all_lora_paths[i])
        #     i = (i + 1) % len(all_lora_paths)

        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=tp_size,
            lora_paths=all_lora_paths,
            max_loras_per_batch=3,
            disable_radix_cache=True,
            # disable_cuda_graph=True,
            trust_remote_code=True,
        ) as srt_runner:
            srt_outputs = srt_runner.batch_forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=tp_size,
            lora_paths=all_lora_paths,
            max_loras_per_batch=3,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            trust_remote_code=True,
        ) as srt_runner:
            srt_outputs_disable_cuda_graph = srt_runner.batch_forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )
        print(f"{srt_outputs.output_strs=}")
        print(f"{srt_outputs_disable_cuda_graph.output_strs=}")
        # for i in range(len(prompts)):
        #     srt_prefill = torch.tensor(srt_outputs.top_input_logprobs[i])
        #     srt_prefill_disable_cuda_graph = torch.tensor(srt_outputs_disable_cuda_graph.top_input_logprobs[i])
        #     max_prefill_diff = torch.max(torch.abs(srt_prefill_disable_cuda_graph - srt_prefill))
        #     print("Max prefill diff:", max_prefill_diff)

        #     # Compare decode stage logprobs
        #     srt_decode = torch.tensor(srt_outputs.top_output_logprobs[i])
        #     srt_decode_disable_cuda_graph = torch.tensor(srt_outputs_disable_cuda_graph.top_output_logprobs[i])
        #     max_decode_diff = torch.max(torch.abs(srt_decode_disable_cuda_graph - srt_decode))
        #     print("Max decode diff:", max_decode_diff)

        for i in range(len(prompts)):
            assert (
                srt_outputs.output_strs[i]
                == srt_outputs_disable_cuda_graph.output_strs[i]
            )

    def test_all(self):
        for lora_set in LORA_SETS:
            # self.load_lora_adapter(lora_set, 1)
            for torch_dtype in TORCH_DTYPES:
                tp_size = 1
                max_new_tokens = 32
                self.inference(PROMPTS, lora_set, tp_size, torch_dtype, max_new_tokens)
                # self.serving(PROMPTS, lora_set, tp_size, torch_dtype, max_new_tokens)
                # self.base_inference(
                #     PROMPTS, lora_set, tp_size, torch_dtype, max_new_tokens
                # )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
