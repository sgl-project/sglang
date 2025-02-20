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
from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
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


def _get_sentence_transformer_embedding_model(model_path, torch_dtype):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import is_sentence_transformer_model

    if is_sentence_transformer_model(model_path):
        model = SentenceTransformer(
            model_path,
            model_kwargs={"torch_dtype": torch_dtype},
        )
    else:  # if no pre-trained sentence-transformers model
        from sentence_transformers import models

        word_embedding_model = models.Transformer(model_path).to(dtype=torch_dtype)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="lasttoken",
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model.cuda()


@dataclass
class ModelOutput:
    output_strs: List[str] = None
    output_ids: List[int] = None
    top_input_logprobs: List[torch.Tensor] = None
    top_output_logprobs: List[torch.Tensor] = None
    embed_logits: List[torch.Tensor] = None
    scores: List[float] = None


class HFRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str = "generation",
        output_str_only: bool = False,
    ):
        self.model_type = model_type
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

    def needs_trust_remote_code(self, model_path):
        models_needs_trust_remote = [
            "LxzGordon/URM-LLaMa-3.1-8B",
        ]
        if model_path in models_needs_trust_remote:
            return True
        return False

    def start_model_process(self, in_queue, out_queue, model_path, torch_dtype):
        # Apply model-specific patches
        monkey_patch_gemma2_sdpa()

        # Load the model and tokenizer
        if self.model_type == "generation":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            ).cuda()
        elif self.model_type == "embedding":
            self.model = _get_sentence_transformer_embedding_model(
                model_path, torch_dtype
            )
        elif self.model_type == "reward":
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.needs_trust_remote_code(model_path),
            ).cuda()
        else:
            raise Exception(f"Unrecognized model type {self.model_type}")
        self.tokenizer = get_tokenizer(model_path, torch_dtype=torch.dtype)

        # Run forward
        while True:
            prompts, max_new_tokens, lora_paths = in_queue.get()
            if lora_paths is not None:
                assert len(prompts) == len(lora_paths)

            if prompts is not None:
                if self.model_type == "generation":
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
                            from peft import PeftModel

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

                        text = self.tokenizer.decode(
                            outputs[0][0][len(input_ids[0]) :], skip_special_tokens=True
                        )
                        # Check if the text is empty or only whitespace.
                        if not text.strip():
                            raise ValueError(
                                "Received an empty text response. Please verify your input or model configuration."
                            )
                        output_strs.append(text)

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

                elif self.model_type == "embedding":
                    assert not self.output_str_only
                    logits = self.model.encode(prompts).tolist()
                    out_queue.put(ModelOutput(embed_logits=logits))

                elif self.model_type == "reward":
                    scores = []
                    for conv in prompts:
                        conv_formatted = self.tokenizer.apply_chat_template(
                            conv, tokenize=False
                        )
                        conv_tokenized = self.tokenizer(
                            conv_formatted, return_tensors="pt"
                        ).to("cuda")
                        scores.append(
                            float(self.model(**conv_tokenized).logits[0][0].item())
                        )
                    out_queue.put(ModelOutput(scores=scores))
                else:
                    raise Exception(f"Unrecognized model type {self.model_type}")

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
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str,
        tp_size: int = 1,
        port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        lora_paths: List[str] = None,
        max_loras_per_batch: int = 4,
        lora_backend: str = "triton",
        disable_cuda_graph: bool = False,
        disable_radix_cache: bool = False,
        mem_fraction_static: float = 0.65,
    ):
        self.model_type = model_type
        self.is_generation = model_type == "generation"
        self.engine = Engine(
            model_path=model_path,
            tp_size=tp_size,
            dtype=get_dtype_str(torch_dtype),
            port=port,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=False,
            is_embedding=not self.is_generation,
            lora_paths=lora_paths,
            max_loras_per_batch=max_loras_per_batch,
            lora_backend=lora_backend,
            disable_cuda_graph=disable_cuda_graph,
            disable_radix_cache=disable_radix_cache,
        )
        self.tokenizer = get_tokenizer(model_path)

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
                response = self.engine.generate(
                    prompt,
                    lora_path=lora_paths[i] if lora_paths else None,
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=0,
                    top_logprobs_num=NUM_TOP_LOGPROBS,
                )
                text = response["text"]

                # Check if the text is empty or only whitespace.
                if not text.strip():
                    raise ValueError(
                        "Received an empty text response. Please verify your input or model configuration."
                    )
                output_strs.append(text)

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
            response = self.engine.encode(prompts)
            if self.model_type == "embedding":
                logits = [x["embedding"] for x in response]
                return ModelOutput(embed_logits=logits)
            else:
                scores = [x["embedding"][0] for x in response]
                return ModelOutput(scores=scores)

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
            response = self.engine.generate(
                prompts,
                lora_path=lora_paths if lora_paths else None,
                sampling_params=sampling_params,
            )
            output_strs = [r["text"] for r in response]

            return ModelOutput(
                output_strs=output_strs,
            )
        else:
            response = self.engine.encode(prompts)
            if self.model_type == "embedding":
                logits = [x["embedding"] for x in response]
                return ModelOutput(embed_logits=logits)
            else:
                scores = [x["embedding"][0] for x in response]
                return ModelOutput(scores=scores)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.shutdown()
        del self.engine


def monkey_patch_gemma2_sdpa():
    """
    Use sdpa by default to fix the OOM issue.
    Revert this commit:
    https://github.com/huggingface/transformers/commit/975b988bfe6e7ebb47390cd9a1556c6888804883#diff-5f76eac6f18f4b491521314c318a9692318feb4d19228e9576cce7bde4240834R660
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2PreTrainedModel

    def _check_and_enable_sdpa(config, hard_check_only: bool = False):
        config._attn_implementation = "sdpa"
        return config

    setattr(Gemma2PreTrainedModel, "_check_and_enable_sdpa", _check_and_enable_sdpa)
