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

import json
import multiprocessing
from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.server import Runtime
from sglang.srt.utils import is_generation_model

DEFAULT_PROMPTS = [
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
    "The capital of the United Kindom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
]

NUM_TOP_LOGPROBS = 5


def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    else:
        raise NotImplementedError()


@dataclass
class ModelOutput:
    output_strs: List[str] = None
    output_ids: List[int] = None
    top_input_logprobs: List[torch.Tensor] = None
    top_output_logprobs: List[torch.Tensor] = None
    embed_logits: List[torch.Tensor] = None


class HFRunner:
    def __init__(
        self,
        model_path,
        torch_dtype=torch.float16,
        is_generation_model=None,
    ):
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()

        self.model_proc = multiprocessing.Process(
            target=self.start_model_process,
            args=(
                self.in_queue,
                self.out_queue,
                model_path,
                torch_dtype,
                is_generation_model,
            ),
        )
        self.model_proc.start()

    def start_model_process(
        self, in_queue, out_queue, model_path, torch_dtype, is_generation_model
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        self.is_generation_model = (
            is_generation_model(model_path)
            if is_generation_model is None
            else is_generation_model
        )
        if self.is_generation_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).cuda()
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_path,
                model_kwargs={"torch_dtype": torch_dtype},
            )

        while True:
            prompts, max_new_tokens = in_queue.get()
            if prompts is not None:
                if self.is_generation_model:
                    output_strs = []
                    prefill_logprobs = []
                    for p in prompts:
                        if isinstance(p, str):
                            input_ids = self.tokenizer.encode(
                                p, return_tensors="pt"
                            ).cuda()
                        else:
                            input_ids = torch.tensor([p], device="cuda")

                        output_ids = self.model.generate(
                            input_ids, do_sample=False, max_new_tokens=max_new_tokens
                        )
                        output_strs.append(
                            self.tokenizer.decode(output_ids[0][len(input_ids[0]) :])
                        )

                        logits = self.model.forward(input_ids).logits[0]
                        logprobs = F.log_softmax(
                            logits, dim=-1, dtype=torch.float32
                        ).tolist()
                        # index_of_max = (lambda nums: nums.index(max(nums)))(logprobs[-1])
                        # print("index", index_of_max)
                        logprobs = [
                            sorted(token_logprobs, reverse=True)[:NUM_TOP_LOGPROBS]
                            for token_logprobs in logprobs
                        ]
                        prefill_logprobs.append(logprobs)

                    out_queue.put(
                        ModelOutput(
                            output_strs=output_strs, top_input_logprobs=prefill_logprobs
                        )
                    )

                else:
                    logits = self.model.encode(prompts).tolist()

                    out_queue.put(ModelOutput(embed_logits=logits))

    def forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=8,
    ):
        self.in_queue.put((prompts, max_new_tokens))
        return self.out_queue.get()

    def terminate(self):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None


class SRTRunner:
    def __init__(
        self,
        model_path,
        tp_size=1,
        torch_dtype=torch.float16,
        is_generation_model=None,
    ):
        self.is_generation_model = (
            is_generation_model(model_path)
            if is_generation_model is None
            else is_generation_model
        )
        self.runtime = Runtime(
            model_path=model_path,
            tp_size=tp_size,
            dtype=get_dtype_str(torch_dtype),
        )

    def forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=8,
    ):
        if self.is_generation_model:
            # the return value contains logprobs from prefill
            output_strs = []
            top_input_logprobs = []
            sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
            for prompt in prompts:
                response = self.runtime.generate(
                    prompt,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    top_logprobs_num=NUM_TOP_LOGPROBS,
                )
                response = json.loads(response)
                output_strs.append(response["text"])
                top_input_logprobs.append(
                    [
                        [tup[0] for tup in x[:NUM_TOP_LOGPROBS]]
                        for x in response["meta_info"]["input_top_logprobs"][1:]
                    ]
                    + [
                        [
                            tup[0]
                            for tup in response["meta_info"]["output_top_logprobs"][0][
                                :NUM_TOP_LOGPROBS
                            ]
                        ]
                    ]
                )

            return ModelOutput(
                output_strs=output_strs, top_input_logprobs=top_input_logprobs
            )
        else:
            response = self.runtime.encode(prompts)
            response = json.loads(response)
            logits = [x["embedding"] for x in response]
            return ModelOutput(embed_logits=logits)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.runtime.shutdown()
        del self.runtime
