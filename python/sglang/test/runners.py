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
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationConfig,
)

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import load_image
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, calculate_rouge_l

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
]
TEST_RERANK_QUERY_DOCS = [
    {
        "query": "How many people live in Berlin?",
        "documents": [
            "Berlin is well known for its museums.",
        ],
    },
    {
        "query": "How many people live in Berlin?",
        "documents": [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin is well known for its museums.",
        ],
    },
]

dirpath = os.path.dirname(__file__)
with open(os.path.join(dirpath, "long_prompt.txt"), "r") as f:
    long_prompt = f.read()
DEFAULT_PROMPTS.append(long_prompt)

NUM_TOP_LOGPROBS = 5


def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.float32:
        return "float32"
    else:
        raise NotImplementedError()


def get_top_logprobs(logits, k):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs, top_indices = torch.topk(logprobs, k=k, dim=-1)
    return logprobs


def get_token_ids_logprobs(logits, token_ids):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs = logprobs[..., token_ids]
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
    top_output_logprob_idx: List[List[int]] = None
    embed_logits: List[torch.Tensor] = None
    scores: List[float] = None
    input_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    output_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    token_ids_input_logprobs: List[torch.Tensor] = None
    token_ids_output_logprobs: List[torch.Tensor] = None


class HFRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str = "generation",
        output_str_only: bool = False,
        trust_remote_code: bool = False,
        patch_model_do_sample_false: bool = False,
    ):
        self.model_type = model_type
        self.output_str_only = output_str_only
        self.trust_remote_code = trust_remote_code
        self.patch_model_do_sample_false = patch_model_do_sample_false

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

    # copy from https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py

    def _get_gme_qwen2_vl_embeddings(
        self, prompts, image_data: Optional[List[str]] = None
    ):

        images = None
        if image_data is not None:
            images = [load_image(image)[0] for image in image_data]

        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=1800,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self._forward_gme_qwen2_vl(**inputs)
        return embeddings.tolist()

    def _forward_gme_qwen2_vl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds = self.model.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.model.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
        )

        embeddings = outputs.hidden_states[-1][:, -1]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def start_model_process(self, in_queue, out_queue, model_path, torch_dtype):
        # Apply model-specific patches
        monkey_patch_gemma2_sdpa()

        # Load the model and tokenizer
        if self.model_type == "generation":
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
            if self.trust_remote_code:
                model_cls = AutoModelForCausalLM
            else:
                model_arch = getattr(config, "architectures")[0]
                model_cls = getattr(transformers, model_arch)
            self.base_model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
            ).cuda()
        elif self.model_type == "embedding":
            if "gme-qwen2-vl" in model_path.lower():
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                ).cuda()
                self.processor = AutoProcessor.from_pretrained(model_path)
            elif "clip" in model_path.lower():
                self.model = AutoModel.from_pretrained(model_path).cuda()
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                self.model = _get_sentence_transformer_embedding_model(
                    model_path, torch_dtype
                )
        elif self.model_type == "reward" or self.model_type == "cross_encoder":
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.needs_trust_remote_code(model_path),
            ).cuda()
        else:
            raise Exception(f"Unrecognized model type {self.model_type}")
        self.tokenizer = get_tokenizer(
            model_path,
            torch_dtype=torch.dtype,
            trust_remote_code=self.trust_remote_code,
        )

        # Run forward
        while True:
            prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob = (
                in_queue.get()
            )
            if lora_paths is not None:
                assert len(prompts) == len(lora_paths)

            if prompts is not None:
                if self.model_type == "generation":
                    out_queue.put(
                        self.forward_generation_raw(
                            base_model=self.base_model,
                            prompts=prompts,
                            max_new_tokens=max_new_tokens,
                            tokenizer=self.tokenizer,
                            lora_paths=lora_paths,
                            torch_dtype=torch_dtype,
                            output_str_only=self.output_str_only,
                            token_ids_logprob=token_ids_logprob,
                            patch_model_do_sample_false=self.patch_model_do_sample_false,
                        )
                    )
                elif self.model_type == "embedding":
                    assert not self.output_str_only
                    if "gme-qwen2-vl" in model_path.lower():
                        logits = self._get_gme_qwen2_vl_embeddings(prompts, image_data)
                    elif "clip" in model_path.lower():
                        if image_data is not None:
                            image = load_image(image_data)
                            inputs = self.processor(
                                images=image[0], return_tensors="pt"
                            )
                            logits = self.model.get_image_features(
                                pixel_values=inputs.data["pixel_values"].cuda(),
                            ).tolist()
                        else:
                            inputs = self.tokenizer(
                                prompts, padding=True, return_tensors="pt"
                            )
                            logits = self.model.get_text_features(
                                input_ids=inputs.data["input_ids"].cuda(),
                                attention_mask=inputs.data["attention_mask"].cuda(),
                            ).tolist()
                    else:
                        logits = self.model.encode(prompts).tolist()
                    out_queue.put(ModelOutput(embed_logits=logits))
                elif self.model_type == "cross_encoder":
                    inputs = self.tokenizer(
                        prompts, padding=True, return_tensors="pt"
                    ).to("cuda")
                    scores = self.model(**inputs).logits
                    scores = scores.squeeze().tolist()
                    if not isinstance(scores, list):
                        scores = [scores]
                    out_queue.put(ModelOutput(scores=scores))

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
        prompts: Union[
            List[List[str]], List[str], List[torch.Tensor]
        ] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        token_ids_logprob: Optional[int] = None,
    ):
        self.in_queue.put(
            (prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob)
        )
        return self.out_queue.get()

    def terminate(self):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    @staticmethod
    def forward_generation_raw(
        base_model,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int,
        tokenizer,
        torch_dtype: torch.dtype,
        lora_paths: Optional[List[str]] = None,
        output_str_only: bool = False,
        token_ids_logprob: Optional[int] = None,
        patch_model_do_sample_false: Optional[bool] = False,
    ) -> ModelOutput:
        output_strs = []
        top_input_logprobs = []
        top_output_logprobs = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None

        for i, p in enumerate(prompts):
            if isinstance(p, str):
                input_ids = tokenizer.encode(p, return_tensors="pt").cuda()
            else:
                input_ids = torch.tensor([p], device="cuda")

            if lora_paths is not None and lora_paths[i] is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    base_model,
                    lora_paths[i],
                    torch_dtype=torch_dtype,
                    is_trainable=False,
                )
            else:
                model = base_model
            if patch_model_do_sample_false:
                model.generation_config.do_sample = False
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=(not output_str_only),
                    # make sure to disable compile
                    disable_compile=True,
                ),
            )

            text = tokenizer.decode(
                outputs[0][0][len(input_ids[0]) :], skip_special_tokens=True
            )
            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)

            if not output_str_only:
                # outputs.scores: (num_token, 1, vocab_size)
                top_output_logprobs.append(
                    [
                        get_top_logprobs(logits[0], NUM_TOP_LOGPROBS).tolist()
                        for logits in outputs.scores
                    ]
                )
                if token_ids_logprob is not None:
                    token_ids_output_logprobs.append(
                        [
                            get_token_ids_logprobs(
                                logits[0], token_ids_logprob
                            ).tolist()
                            for logits in outputs.scores
                        ]
                    )
                del outputs

                input_logits = model.forward(input_ids).logits[0]
                top_input_logprobs.append(
                    get_top_logprobs(input_logits, NUM_TOP_LOGPROBS).tolist()
                )
                if token_ids_logprob is not None:
                    token_ids_input_logprobs.append(
                        get_token_ids_logprobs(input_logits, token_ids_logprob).tolist()
                    )
                del input_logits

            if lora_paths is not None and lora_paths[i] is not None:
                # Unload the LoRA adapter if it is used
                model.unload()

        return ModelOutput(
            output_strs=output_strs,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
        )


class SRTRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str,
        tp_size: int = 1,
        model_impl: str = "auto",
        port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        lora_paths: Optional[Union[List[str], List[dict[str, str]]]] = None,
        max_loras_per_batch: int = 4,
        attention_backend: Optional[str] = None,
        prefill_attention_backend: Optional[str] = None,
        decode_attention_backend: Optional[str] = None,
        lora_backend: str = "triton",
        disable_cuda_graph: bool = False,
        disable_radix_cache: bool = False,
        chunked_prefill_size: Optional[int] = None,
        dp_size: int = 1,
        tokenizer_path: Optional[str] = None,
        mem_fraction_static: float = 0.65,
        trust_remote_code: bool = False,
        speculative_draft_model_path: Optional[str] = None,
        speculative_draft_model_revision: Optional[str] = None,
        speculative_algorithm: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        disable_overlap_schedule: bool = False,
        disable_custom_all_reduce: bool = False,
        torchao_config: Optional[str] = None,
        cuda_graph_max_bs: int = 4,
        sleep_on_idle=False,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        enable_lora: Optional[bool] = None,
        max_loaded_loras: Optional[int] = None,
        lora_eviction_policy: str = "lru",
    ):
        self.model_type = model_type
        self.is_generation = model_type == "generation"
        enable_dp_attention = dp_size > 1

        spec_kwargs = {}
        if speculative_draft_model_path:
            spec_kwargs["speculative_draft_model_path"] = speculative_draft_model_path
            spec_kwargs["speculative_draft_model_revision"] = (
                speculative_draft_model_revision
            )
            spec_kwargs["speculative_algorithm"] = speculative_algorithm
            spec_kwargs["speculative_num_steps"] = speculative_num_steps
            spec_kwargs["speculative_eagle_topk"] = speculative_eagle_topk
            spec_kwargs["speculative_num_draft_tokens"] = speculative_num_draft_tokens

        self.engine = Engine(
            model_path=model_path,
            tp_size=tp_size,
            dtype=get_dtype_str(torch_dtype),
            port=port,
            model_impl=model_impl,
            torchao_config=torchao_config,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=trust_remote_code,
            is_embedding=not self.is_generation,
            lora_paths=lora_paths,
            max_loras_per_batch=max_loras_per_batch,
            lora_backend=lora_backend,
            attention_backend=attention_backend,
            prefill_attention_backend=prefill_attention_backend,
            decode_attention_backend=decode_attention_backend,
            disable_cuda_graph=disable_cuda_graph,
            disable_radix_cache=disable_radix_cache,
            chunked_prefill_size=chunked_prefill_size,
            enable_dp_attention=enable_dp_attention,
            dp_size=dp_size,
            tokenizer_path=tokenizer_path,
            disable_overlap_schedule=disable_overlap_schedule,
            cuda_graph_max_bs=cuda_graph_max_bs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            sleep_on_idle=sleep_on_idle,
            max_lora_rank=max_lora_rank,
            lora_target_modules=lora_target_modules,
            enable_lora=enable_lora,
            max_loaded_loras=max_loaded_loras,
            lora_eviction_policy=lora_eviction_policy,
            **spec_kwargs,
        )

        if tokenizer_path is None:
            self.tokenizer = get_tokenizer(
                model_path, trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = None

    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        return self.engine.load_lora_adapter(lora_name, lora_path, pinned)

    def unload_lora_adapter(self, lora_name: str):
        return self.engine.unload_lora_adapter(lora_name)

    def forward(
        self,
        prompts: Union[
            List[List[str]], List[str], List[torch.Tensor]
        ] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        logprob_start_len: int = 0,
        top_k: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
    ):
        if self.is_generation:
            return self.forward_generation_raw(
                engine=self.engine,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                lora_paths=lora_paths,
                logprob_start_len=logprob_start_len,
                top_k=top_k,
                token_ids_logprob=token_ids_logprob,
            )
        else:
            if self.model_type == "embedding":
                response = self.engine.encode(prompt=prompts, image_data=image_data)
                if isinstance(response, list):
                    logits = [x["embedding"] for x in response]
                else:
                    logits = [response["embedding"]]
                return ModelOutput(embed_logits=logits)
            # cross encoder model
            elif self.model_type == "cross_encoder":
                response = self.engine.rerank(prompts)
                if not isinstance(response, list):
                    response = [response]
                scores = [x["embedding"] for x in response]
                return ModelOutput(scores=scores)
            # reward model
            else:
                response = self.engine.encode(prompts)
                scores = [x["embedding"][0] for x in response]
                return ModelOutput(scores=scores)

    def batch_forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens=8,
        lora_paths=None,
    ):
        """
        testing serving by sending all prompts once
        only return output strings and no logprobs
        """
        if self.is_generation:
            return self.batch_forward_generation_raw(
                engine=self.engine,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                lora_paths=lora_paths,
            )
        else:
            response = self.engine.encode(prompts, image_data)
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

    @staticmethod
    def forward_generation_raw(
        engine: Engine,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        logprob_start_len: int = 0,
        top_k: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        output_ids = []
        # Input logprobs. Note that the last item in input logprob is equivalent to
        # the first item in the output logprob.
        top_input_logprobs = []
        input_token_logprobs_lst = []
        top_output_logprobs = []
        output_token_logprobs_lst = []
        top_output_logprob_idx = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None

        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        if top_k:
            sampling_params["top_k"] = top_k

        for i, prompt in enumerate(prompts):
            response = engine.generate(
                prompt,
                lora_path=lora_paths[i] if lora_paths else None,
                sampling_params=sampling_params,
                return_logprob=True,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=NUM_TOP_LOGPROBS,
                token_ids_logprob=token_ids_logprob,
            )
            text = response["text"]

            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)
            # output_ids.append(response["output_ids"])

            input_token_logprobs = response["meta_info"]["input_token_logprobs"]
            output_token_logprobs = response["meta_info"]["output_token_logprobs"]
            # print(i, input_token_logprobs)
            # print(i, output_token_logprobs)
            logprobs = response["meta_info"]["input_top_logprobs"]
            if token_ids_logprob is not None:
                input_token_ids_logprobs = response["meta_info"][
                    "input_token_ids_logprobs"
                ][1:]
            else:
                input_token_ids_logprobs = None

            num_prompt_tokens = response["meta_info"]["prompt_tokens"]
            assert len(input_token_logprobs) == num_prompt_tokens - logprob_start_len
            assert len(logprobs) == num_prompt_tokens - logprob_start_len

            # The first token logprob has no meaning in sglang.
            input_token_logprobs = input_token_logprobs[1:]
            logprobs = logprobs[1:]
            assert len(input_token_logprobs) == len(logprobs)

            input_token_logprobs_lst.append(
                input_token_logprobs + [output_token_logprobs[0]]
            )
            output_token_logprobs_lst.append(output_token_logprobs)

            top_input_logprobs.append(
                [[tup[0] for tup in x[:NUM_TOP_LOGPROBS]] for x in logprobs]
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
            top_output_logprob_idx.append(
                [
                    [tup[1] for tup in x[:NUM_TOP_LOGPROBS]]
                    for x in response["meta_info"]["output_top_logprobs"]
                ]
            )
            if token_ids_logprob is not None:
                token_ids_input_logprobs.append(
                    [[tup[0] for tup in x] for x in input_token_ids_logprobs]
                    + [
                        [
                            tup[0]
                            for tup in response["meta_info"][
                                "output_token_ids_logprobs"
                            ][0]
                        ]
                    ]
                )
                token_ids_output_logprobs.append(
                    [
                        [tup[0] for tup in x]
                        for x in response["meta_info"]["output_token_ids_logprobs"]
                    ]
                )

        return ModelOutput(
            output_strs=output_strs,
            output_ids=output_ids,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            input_token_logprobs_lst=input_token_logprobs_lst,
            output_token_logprobs_lst=output_token_logprobs_lst,
            top_output_logprob_idx=top_output_logprob_idx,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
        )

    @staticmethod
    def batch_forward_generation_raw(
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens,
        lora_paths,
        engine,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        response = engine.generate(
            prompts,
            lora_path=lora_paths if lora_paths else None,
            sampling_params=sampling_params,
        )
        output_strs = [r["text"] for r in response]

        return ModelOutput(
            output_strs=output_strs,
        )


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


def check_close_model_outputs(
    hf_outputs: ModelOutput,
    srt_outputs: ModelOutput,
    prefill_tolerance: float,
    decode_tolerance: float,
    rouge_l_tolerance: float,
    debug_text: str = "",
    check_logprobs: bool = True,
):
    # Compare output strings
    print(f"{hf_outputs.output_strs=}")
    print(f"{srt_outputs.output_strs=}")
    rouge_l_scores = calculate_rouge_l(hf_outputs.output_strs, srt_outputs.output_strs)
    print(f"{rouge_l_scores=}")
    assert all(
        score >= rouge_l_tolerance for score in rouge_l_scores
    ), f"Not all ROUGE-L scores are greater than rouge_l_tolerance={rouge_l_tolerance}"

    if check_logprobs:
        for i in range(len(hf_outputs.output_strs)):
            # Compare input logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])
            input_len = hf_logprobs.shape[0]
            print(
                "prefill logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(abs(hf_logprobs - srt_logprobs) < prefill_tolerance), (
                    f"prefill logprobs are not all close with {debug_text} "
                    f"prefill_tolerance={prefill_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )

            # Compare output logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_output_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_output_logprobs[i])

            print(
                "decode logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(abs(hf_logprobs - srt_logprobs) < decode_tolerance), (
                    f"decode logprobs are not all close with {debug_text} "
                    f"decode_tolerance={decode_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )
