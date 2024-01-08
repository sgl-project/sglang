import functools
from enum import Enum, auto
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import transformers
from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.lang.interpreter import ProgramState
from sglang.utils import get_available_gpu_memory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformersgl.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class StopReason(Enum):
    EOS_TOKEN = auto()
    STOP_STR = auto()
    LENGTH = auto()


def load_model(
    model_name: str,
    device,
    num_gpus,
    max_gpu_memory,
    model_kwargs=None,
    tokenizer_kwargs=None,
):
    model_kwargs = model_kwargs or {}
    tokenizer_kwargs = tokenizer_kwargs or {}

    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        if num_gpus != 1:
            model_kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                model_kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = [
                    get_available_gpu_memory(i, False) for i in range(num_gpus)
                ]
                model_kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                model_kwargs["max_memory"] = {
                    i: max_gpu_memory for i in range(num_gpus)
                }
    elif device == "cpu":
        model_kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"Invalid device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    if num_gpus == 1:
        model.to(device).eval()

    return model, tokenizer


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@functools.lru_cache
def get_token_healing_mask(tokenizer, prompt_last_token):
    last_str = tokenizer.convert_ids_to_tokens(prompt_last_token)
    disallowed = torch.zeros(len(tokenizer), dtype=bool)
    for s, t_id in tokenizer.get_vocab().items():
        if not s.startswith(last_str):
            disallowed[t_id] = 1
    return disallowed


@functools.lru_cache
def get_int_token_mask(tokenizer):
    disallowed = torch.zeros(len(tokenizer), dtype=bool)
    for s, t_id in tokenizer.get_vocab().items():
        s = s.replace("▁", "").strip()
        if not (s.isdigit() or len(s) == 0 or s == ","):
            disallowed[t_id] = 1
    disallowed[tokenizer.eos_token_id] = 0
    return disallowed


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    stop: List[str],
    temperature,
    top_p,
    token_healing,
    logit_mask=None,
):
    logits_processor = prepare_logits_processor(
        temperature=temperature, repetition_penalty=1.0, top_p=top_p, top_k=0
    )
    device = model.device
    input_ids = tokenizer.encode(prompt)
    output_ids = list(input_ids)
    prompt_len = len(prompt)

    # Resolve stop
    stop_token_ids = [tokenizer.eos_token_id]

    # Token healing
    token_healing = token_healing and len(input_ids) > 0
    if token_healing:
        token_healing_mask = get_token_healing_mask(tokenizer, input_ids[-1])
        del output_ids[-1]

    # Generate
    past_key_values = None
    stop_reason = None
    for i in range(max_new_tokens):
        # Forward
        if i == 0:  # prefill
            out = model(torch.as_tensor([output_ids], device=device), use_cache=True)
        else:  # decoding
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
        logits = out.logits
        past_key_values = out.past_key_values

        # Logit mask
        if token_healing and i == 0:
            logits[0, -1, token_healing_mask] = -1e4
        if logit_mask is not None:
            logits[0, -1, logit_mask] = -1e4

        # Sample next token
        last_token_logits = logits_processor(None, logits[:, -1, :])[0]
        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
        output_ids.append(token)

        # Stop condition
        if token in stop_token_ids:
            stop_reason = StopReason.EOS_TOKEN
            break

        output_str = tokenizer.decode(output_ids, skip_special_tokens=True)
        for stop_str in stop:
            pos = output_str[prompt_len:].find(stop_str)
            if pos != -1:
                stop_reason = StopReason.STOP_STR
                output_str = output_str[: prompt_len + pos]
                break

        if stop_reason:
            break

    return output_str[prompt_len:]


class HuggingFaceTransformers(BaseBackend):
    def __init__(
        self,
        model_name,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        model_kwargs=None,
        tokenizer_kwargs=None,
    ):
        self.model_name = model_name
        self.device = device

        self.model, self.tokenizer = load_model(
            model_name, device, num_gpus, max_gpu_memory, model_kwargs, tokenizer_kwargs
        )

        self.chat_template = get_chat_template_by_model_path(model_name)

    def get_chat_template(self):
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        pass

    def uncache_prefix(self, rid: str):
        pass

    def end_request(self, rid: str):
        pass

    def begin_program(self, s: ProgramState):
        pass

    def end_program(self, s: ProgramState):
        pass

    def fill(self, s: ProgramState, text: str):
        return False

    def generate_internal(
        self,
        prompt: str,
        max_tokens: int,
        stop: Union[str, List[str]],
        temperature: float,
        top_p: float,
        dtype: Optional[str] = None,
    ):
        if dtype is None:
            comp = generate_stream(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                token_healing=True,
            )
        elif dtype in [str, "str", "string"]:
            comp = generate_stream(
                self.model,
                self.tokenizer,
                prompt + '"',
                max_new_tokens=max_tokens,
                stop=['"'],
                temperature=temperature,
                top_p=top_p,
                token_healing=False,
            )
            comp = '"' + comp + '"'
        elif dtype in [int, "int"]:
            logit_mask = get_int_token_mask(self.tokenizer)
            comp = generate_stream(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                stop=stop + [" ", ","],
                temperature=temperature,
                top_p=top_p,
                token_healing=False,
                logit_mask=logit_mask,
            )
        return comp

    def generate(
        self,
        s: ProgramState,
        max_tokens: int,
        stop: Union[str, List[str]],
        temperature: float,
        top_p: float,
        dtype: Optional[str] = None,
    ):
        prompt = s.text
        comp = self.generate_internal(
            prompt, max_tokens, stop, temperature, top_p, dtype
        )
        return comp

    def parallel_generate(
        self,
        s: ProgramState,
        prefixes: List[str],
        join_func: Callable,
        max_tokens: int,
        stop: Union[str, List[str]],
        temperature: float,
        top_p: float,
        dtype: Optional[str] = None,
    ):
        prompt = s.text
        parallel_prompts = [prompt + prefix for prefix in prefixes]

        comps = []
        for i in range(len(parallel_prompts)):
            comps.append(
                self.generate_internal(
                    parallel_prompts[i], max_tokens, stop, temperature, top_p, dtype
                )
            )

        joined = join_func([p + c for p, c in zip(prefixes, comps)])
        return joined, comps

    @torch.inference_mode()
    def select(
        self, s: ProgramState, choices: List[str], temperature: float, top_p: float
    ):
        loss_fct = torch.nn.CrossEntropyLoss()
        prompt = s.text

        prompt_len = self.tokenizer.encode(prompt, return_tensors="pt").shape[1]
        prompt_choices = [prompt + choice for choice in choices]

        scores = []
        for i in range(len(choices)):
            choice_ids = self.tokenizer.encode(
                prompt_choices[i], return_tensors="pt"
            ).to(self.model.device)
            logits = self.model(choice_ids).logits

            # score = -loss_fct(logits[0, :-1, :], choice_ids[0, 1:]).item()

            logprobs = torch.log(torch.softmax(logits, dim=-1))
            idx1 = torch.arange(0, logits.shape[1] - 1, device=logits.device)
            idx2 = choice_ids[0, 1:]
            selected_logprobs = logprobs[0, idx1, idx2]
            score = selected_logprobs.mean().item()
            scores.append(score)

        decision = choices[np.argmax(scores)]
        return decision, scores
