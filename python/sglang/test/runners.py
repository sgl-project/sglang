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
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.server import Runtime
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER

DEFAULT_PROMPTS = [
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
]

dirpath = os.path.dirname(__file__)
with open(os.path.join(dirpath, "long_prompt.txt"), "r") as f:
    long_prompt = f.read()
DEFAULT_PROMPTS.append(long_prompt)

NUM_TOP_LOGPROBS = 5


def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    else:
        raise NotImplementedError()


def get_top_logprobs(logits, k):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs, top_indices = torch.topk(logprobs, k=k, dim=-1)
    return logprobs


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
        torch_dtype,
        is_generation,
        output_str_only=False,
    ):
        self.is_generation = is_generation
        self.output_str_only = output_str_only

        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()

        self.model_proc = mp.Process(
            target=self.start_model_process,
            args=(
                self.in_queue,
                self.out_queue,
                model_path,
                torch_dtype,
            ),
        )
        self.model_proc.start()

    def start_model_process(self, in_queue, out_queue, model_path, torch_dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )

        if self.is_generation:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            ).cuda()
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_path,
                model_kwargs={"torch_dtype": torch_dtype},
            )

        while True:
            prompts, max_new_tokens, lora_paths = in_queue.get()
            if lora_paths is not None:
                assert len(prompts) == len(lora_paths)

            if prompts is not None:
                if self.is_generation:
                    output_strs = []
                    top_input_logprobs = []
                    top_output_logprobs = []
                    for i, p in enumerate(prompts):
                        if isinstance(p, str):
                            input_ids = self.tokenizer.encode(
                                p, return_tensors="pt"
                            ).cuda()
                        else:
                            input_ids = torch.tensor([p], device="cuda")

                        if lora_paths is not None and lora_paths[i] is not None:
                            self.model = PeftModel.from_pretrained(
                                self.base_model,
                                lora_paths[i],
                                torch_dtype=torch_dtype,
                                is_trainable=False,
                            )
                        else:
                            self.model = self.base_model

                        outputs = self.model.generate(
                            input_ids,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            max_new_tokens=max_new_tokens,
                            return_dict_in_generate=True,
                            output_scores=(not self.output_str_only),
                        )
                        output_strs.append(
                            self.tokenizer.decode(outputs[0][0][len(input_ids[0]) :])
                        )
                        if not self.output_str_only:
                            # outputs.scores: (num_token, 1, vocab_size)
                            top_output_logprobs.append(
                                [
                                    get_top_logprobs(
                                        logits[0], NUM_TOP_LOGPROBS
                                    ).tolist()
                                    for logits in outputs.scores
                                ]
                            )
                            del outputs

                            input_logits = self.model.forward(input_ids).logits[0]
                            top_input_logprobs.append(
                                get_top_logprobs(
                                    input_logits, NUM_TOP_LOGPROBS
                                ).tolist()
                            )
                            del input_logits

                    out_queue.put(
                        ModelOutput(
                            output_strs=output_strs,
                            top_input_logprobs=top_input_logprobs,
                            top_output_logprobs=top_output_logprobs,
                        )
                    )

                else:
                    assert not self.output_str_only
                    logits = self.model.encode(prompts).tolist()
                    out_queue.put(ModelOutput(embed_logits=logits))

    def forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=8,
        lora_paths=None,
    ):
        self.in_queue.put((prompts, max_new_tokens, lora_paths))
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
        torch_dtype,
        is_generation,
        tp_size=1,
        port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        lora_paths=None,
        max_loras_per_batch=4,
        disable_cuda_graph=False,
        disable_radix_cache=False,
    ):
        self.is_generation = is_generation
        self.runtime = Runtime(
            model_path=model_path,
            tp_size=tp_size,
            dtype=get_dtype_str(torch_dtype),
            port=port,
            mem_fraction_static=0.69,
            trust_remote_code=False,
            is_embedding=not self.is_generation,
            lora_paths=lora_paths,
            max_loras_per_batch=max_loras_per_batch,
            disable_cuda_graph=disable_cuda_graph,
            disable_radix_cache=disable_radix_cache,
        )

    def forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=8,
        lora_paths=None,
    ):
        if self.is_generation:
            # the return value contains logprobs from prefill
            output_strs = []
            top_input_logprobs = []
            top_output_logprobs = []
            sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
            for i, prompt in enumerate(prompts):
                response = self.runtime.generate(
                    prompt,
                    lora_path=lora_paths[i] if lora_paths else None,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=0,
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
                top_output_logprobs.append(
                    [
                        [tup[0] for tup in x[:NUM_TOP_LOGPROBS]]
                        for x in response["meta_info"]["output_top_logprobs"]
                    ]
                )

            return ModelOutput(
                output_strs=output_strs,
                top_input_logprobs=top_input_logprobs,
                top_output_logprobs=top_output_logprobs,
            )
        else:
            response = self.runtime.encode(prompts)
            response = json.loads(response)
            logits = [x["embedding"] for x in response]
            return ModelOutput(embed_logits=logits)

    def batch_forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=8,
        lora_paths=None,
    ):
        """
        testing serving by sending all prompts once
        only return output strings and no logprobs
        """
        if self.is_generation:
            # the return value contains logprobs from prefill
            output_strs = []
            sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
            response = self.runtime.generate(
                prompts,
                lora_path=lora_paths if lora_paths else None,
                sampling_params=sampling_params,
            )
            response = json.loads(response)
            output_strs = [r["text"] for r in response]

            return ModelOutput(
                output_strs=output_strs,
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
