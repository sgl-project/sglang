from typing import Callable, List, Optional, Union

import numpy as np
from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

try:
    import openai
    import tiktoken
except ImportError as e:
    openai = tiktoken = e


def create_logit_bias_int(tokenizer):
    """Get logit bias for integer numbers."""
    int_token_ids = []

    tokens = tokenizer._mergeable_ranks
    for token, token_id in tokens.items():
        s = tokenizer.decode([token_id])
        if all([c.isdigit() for c in s]) or s in [" "]:
            int_token_ids.append(token_id)
            if len(int_token_ids) >= 300:  # OpenAI API limit
                break
    special_tokens = tokenizer._special_tokens
    mask = {t: 100 for t in int_token_ids[:299]}
    mask[special_tokens["<|endoftext|>"]] = 100
    return mask


CHAT_MODEL_NAMES = [
    # GPT-4
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-0613",
    "gpt-4-0314",
    # GPT-3.5
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
]


class OpenAI(BaseBackend):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()
        self.client = openai.OpenAI(*args, **kwargs)

        if isinstance(openai, Exception):
            raise e

        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.logit_bias_int = create_logit_bias_int(self.tokenizer)

        if model_name in CHAT_MODEL_NAMES:
            self.is_chat_model = True
        else:
            self.is_chat_model = False

        self.chat_template = get_chat_template("default")

    def get_chat_template(self):
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if sampling_params.dtype is None:
            if self.is_chat_model:
                assert s.text_.endswith("ASSISTANT:")
                prompt = s.messages_
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            comp = openai_completion(
                client=self.client,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
        elif sampling_params.dtype in [str, "str", "string"]:
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = openai_completion(
                client=self.client,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_ + '"',
                stop='"',
                **kwargs,
            )
            comp = '"' + comp + '"'
        elif sampling_params.dtype in [int, "int"]:
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = openai_completion(
                client=self.client,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_,
                logit_bias=self.logit_bias_int,
                stop=[" "],
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if sampling_params.dtype is None:
            if self.is_chat_model:
                assert s.text_.endswith("ASSISTANT:")
                prompt = s.messages_
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            generator = openai_completion_stream(
                client=self.client,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
            return generator
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
    ):
        n_choices = len(choices)
        token_ids = [self.tokenizer.encode(x) for x in choices]
        scores = [0] * n_choices
        valid = [len(x) > 0 for x in token_ids]
        prompt_tokens = self.tokenizer.encode(s.text_)

        max_len = max([len(x) for x in token_ids])
        for step in range(max_len):
            # Build logit bias
            logit_bias = {}
            for i in range(n_choices):
                if valid[i]:
                    logit_bias[token_ids[i][step]] = 100

            # Call API
            ret = self.client.completions.create(
                model=self.model_name,
                prompt=prompt_tokens,
                logit_bias=logit_bias,
                max_tokens=1,
                temperature=temperature,
            )
            ret_str = ret.choices[0].text
            ret_token = self.tokenizer.encode(ret_str)[0]

            # TODO:
            # 1. return logits as the scores
            # 2. compute logits of the full choice
            # 3. consider chunk-based decoding

            # Update valid
            hit = False
            for i in range(n_choices):
                if valid[i]:
                    if step == len(token_ids[i]) - 1:
                        valid[i] = False

                    if ret_token == token_ids[i][step]:
                        scores[i] += 1
                        hit = True
                    else:
                        valid[i] = False
            assert hit

            if np.sum(valid) <= 1:
                break

            prompt_tokens.append(ret_token)

        decision = choices[np.argmax(scores)]
        return decision, scores


def openai_completion(client, is_chat=None, prompt=None, **kwargs):
    try:
        if is_chat:
            if kwargs["stop"] is None:
                kwargs.pop("stop")
            ret = client.chat.completions.create(messages=prompt, **kwargs)
            comp = ret.choices[0].message.content
        else:
            ret = client.completions.create(prompt=prompt, **kwargs)
            if isinstance(prompt, (list, tuple)):
                comp = [c.text for c in ret.choices]
            else:
                comp = ret.choices[0].text
    except openai.OpenAIError as e:
        print(f"OpenAI Error: {e}")
        raise e

    return comp


def openai_completion_stream(client, is_chat=None, prompt=None, **kwargs):
    try:
        if is_chat:
            generator = client.chat.completions.create(
                messages=prompt, stream=True, **kwargs
            )
            for ret in generator:
                content = ret.choices[0].delta.content
                yield content or "", {}
        else:
            generator = client.completions.create(prompt=prompt, stream=True, **kwargs)
            for ret in generator:
                content = ret.choices[0].text
                yield content or "", {}
    except openai.OpenAIError as e:
        print(f"OpenAI Error: {e}")
        raise e
