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
    {"base": "meta-llama/Llama-2-7b-hf", "loras": ["winddude/wizardLM-LlaMA-LoRA-7B"]},
    # {"base": "Qwen/Qwen2.5-14B-Instruct", "loras": ["mssongit/Qwen2.5-14B-SFT-LoRA"]},
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
Tell me about llamas and alpacas
### Response:
Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
### Question 2:
What do you know about llamas?
### Answer:
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
        batch_lora_paths = [None]
        i = 0
        for _ in range(len(prompts) - 1):
            batch_lora_paths.append(all_lora_paths[i])
            i = (i + 1) % len(all_lora_paths)

        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=tp_size,
            lora_paths=all_lora_paths,
            max_loras_per_batch=3,
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

        with HFRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
        ) as hf_runner:
            hf_no_lora_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens
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

        for i in range(len(prompts)):
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

            # compare output logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_output_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_output_logprobs[i])
            # print(
            #     "\noutput logprobs diff",
            #     [
            #         float(torch.max(abs(hf_logprobs[j] - srt_logprobs[j])))
            #         for j in range(max_new_tokens)
            #     ],
            # )
            print(
                "max output diff between hf_lora and srt_lora",
                torch.max(abs(hf_logprobs - srt_logprobs)),
                "\n",
            )

        # compare output strings
        print(f"{hf_outputs.output_strs=}")
        print(f"{srt_outputs.output_strs=}")
        print(f"{hf_no_lora_outputs.output_strs=}")
        print(f"{srt_no_lora_outputs.output_strs=}")
        for i in range(len(prompts)):
            assert srt_outputs.output_strs[i].strip(" ") == hf_outputs.output_strs[i], (
                srt_outputs.output_strs[i].strip(" "),
                hf_outputs.output_strs[i],
            )
            # assert (
            #     srt_no_lora_outputs.output_strs[i].strip(" ")
            #     == hf_no_lora_outputs.output_strs[i]
            # ), (
            #     srt_no_lora_outputs.output_strs[i].strip(" "),
            #     hf_no_lora_outputs.output_strs[i],
            # )

    def serving(self, prompts, lora_set, tp_size, torch_dtype, max_new_tokens):
        print("=================== testing serving =======================")
        # test batch forward
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = [None]
        i = 0
        for _ in range(len(prompts) - 1):
            batch_lora_paths.append(all_lora_paths[i])
            i = (i + 1) % len(all_lora_paths)

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=all_lora_paths,
            max_loras_per_batch=3,
            disable_cuda_graph=True,
            disable_radix_cache=True,
        ) as srt_runner:
            srt_outputs = srt_runner.batch_forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        with HFRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            output_str_only=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        # compare output strings
        print(f"{hf_outputs.output_strs=}")
        print(f"{srt_outputs.output_strs=}")
        for i in range(len(prompts)):
            assert srt_outputs.output_strs[i].strip(" ") == hf_outputs.output_strs[i], (
                srt_outputs.output_strs[i].strip(" "),
                hf_outputs.output_strs[i],
            )

    def base_inference(self, prompts, lora_set, tp_size, torch_dtype, max_new_tokens):
        print("=================== testing base inference =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = [None] * len(prompts)

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
        ) as srt_runner:
            srt_no_lora_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        with SRTRunner(
            base_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=all_lora_paths,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=batch_lora_paths
            )

        for i in range(len(prompts)):
            srt_no_lora_logprobs = torch.Tensor(
                srt_no_lora_outputs.top_input_logprobs[i]
            )
            srt_logprobs = torch.uensor(srt_outputs.top_input_logprobs[i])
            print("max_diff", torch.max(abs(srt_no_lora_logprobs - srt_logprobs)))

        print(f"{srt_no_lora_outputs.output_strs=}")
        print(f"{srt_outputs.output_strs=}")

        for i in range(len(prompts)):
            assert srt_outputs.output_strs[i].strip(" ") == hf_outputs.output_strs[i], (
                srt_outputs.output_strs[i].strip(" "),
                hf_outputs.output_strs[i],
            )
            assert (
                srt_no_lora_outputs[i].output_strs.strip(" ")
                == hf_no_lora_outputs[i].output_strs
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
