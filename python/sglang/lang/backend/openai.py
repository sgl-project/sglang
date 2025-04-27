import dataclasses
import logging
import time
import warnings
from typing import List, Optional, Union

import numpy as np

from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import ChatTemplate, get_chat_template_by_model_path
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

try:
    import openai
    import tiktoken
except ImportError as e:
    openai = tiktoken = e


logger = logging.getLogger(__name__)


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


INSTRUCT_MODEL_NAMES = [
    "gpt-3.5-turbo-instruct",
]


@dataclasses.dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    def reset(self):
        self.prompt_tokens = self.completion_tokens = 0


class OpenAI(BaseBackend):
    def __init__(
        self,
        model_name: str,
        is_chat_model: Optional[bool] = None,
        chat_template: Optional[ChatTemplate] = None,
        is_azure: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        if isinstance(openai, Exception):
            raise openai

        if is_azure:
            self.client = openai.AzureOpenAI(*args, **kwargs)
        else:
            self.client = openai.OpenAI(*args, **kwargs)

        self.model_name = model_name
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.logit_bias_int = create_logit_bias_int(self.tokenizer)

        self.chat_template = chat_template or get_chat_template_by_model_path(
            model_name
        )

        if is_chat_model is not None:
            self.is_chat_model = is_chat_model
        else:
            if model_name in INSTRUCT_MODEL_NAMES:
                self.is_chat_model = False
            else:
                self.is_chat_model = True

        self.chat_prefix = self.chat_template.role_prefix_and_suffix["assistant"][0]

        # Usage
        self.token_usage = TokenUsage(0, 0)

        # API speculative execution
        # TODO(ying): This does not support multi-threading (run_batch)
        self.spec_kwargs = {}
        self.spec_format = []
        self.spec_max_num_tries = 3

    def get_chat_template(self):
        return self.chat_template

    def _prepare_spec_execution(
        self,
        sampling_params: SglSamplingParams,
        num_api_spec_tokens: int,
        spec_var_name: str,
    ):
        if "max_tokens" not in self.spec_kwargs:
            self.spec_kwargs["max_tokens"] = num_api_spec_tokens
        else:
            assert self.spec_kwargs["max_tokens"] == num_api_spec_tokens

        params = sampling_params.to_openai_kwargs()
        for key, value in params.items():
            if key in ["stop"]:
                continue
            if key in ["max_tokens"]:
                warnings.warn(
                    "The parameter max_tokens will be overwritten by speculated number of tokens."
                )
                continue
            if key not in self.spec_kwargs:
                self.spec_kwargs[key] = value
            else:
                assert (
                    value == self.spec_kwargs[key]
                ), "sampling parameters should be consistent if turn on api speculative execution."
        self.spec_format.append(
            {"text": "", "stop": params["stop"], "name": spec_var_name}
        )
        return "", {}

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
        spec_var_name: str = None,
    ):
        if sampling_params.dtype is None:
            if self.is_chat_model:
                if s.num_api_spec_tokens is None:
                    if not s.text_.endswith(self.chat_prefix):
                        raise RuntimeError(
                            "This use case is not supported if api speculative execution is off. "
                            "For OpenAI chat models, sgl.gen must be right after sgl.assistant. "
                            "Example of adding api speculative execution: @function(num_api_spec_tokens=128)."
                        )
                    prompt = s.messages_
                else:
                    return self._prepare_spec_execution(
                        sampling_params, s.num_api_spec_tokens, spec_var_name
                    )
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            if (
                self.model_name.startswith("o1")
                or self.model_name.startswith("o3")
                or "o1" in self.model_name
            ):
                kwargs.pop("max_tokens", None)
            else:
                kwargs.pop("max_completion_tokens", None)

            comp = openai_completion(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
            # Keep the returned list (or string) as is.
        elif sampling_params.dtype in [str, "str", "string"]:
            assert (
                not self.is_chat_model
            ), "constrained type not supported on chat model"
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = openai_completion(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_ + '"',
                stop='"',
                **kwargs,
            )
            # Wrap each element in quotes if we have a list.
            if isinstance(comp, list):
                comp = ['"' + x + '"' for x in comp]
            else:
                comp = '"' + comp + '"'
        elif sampling_params.dtype in [int, "int"]:
            assert (
                not self.is_chat_model
            ), "constrained type not supported on chat model"
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = openai_completion(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_,
                logit_bias=self.logit_bias_int,
                stop=[" "],
                **kwargs,
            )
            # Leave as a list if that's what is returned.
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")

        return comp, {}

    def spec_fill(self, value: str):
        assert self.is_chat_model
        self.spec_format.append({"text": value, "stop": None, "name": None})

    def spec_pattern_match(self, comp):
        for i, term in enumerate(self.spec_format):
            text = term["text"]
            if text != "":
                if comp.startswith(text):
                    comp = comp[len(text) :]
                else:
                    return False
            else:
                pos = comp.find(term["stop"])
                if pos != -1:
                    term["text"] = comp[:pos]
                    comp = comp[pos:]
                else:
                    if i == len(self.spec_format) - 1:
                        term["text"] = comp
                    else:
                        return False
        return True

    def role_end_generate(
        self,
        s: StreamExecutor,
    ):
        if s.num_api_spec_tokens is None or not s.text_.endswith(self.chat_prefix):
            return

        comp = ""
        if not all(x["name"] is None for x in self.spec_format):
            # TODO(ying): throw errors or warnings
            for i in range(self.spec_max_num_tries):
                comp = openai_completion(
                    client=self.client,
                    token_usage=self.token_usage,
                    is_chat=self.is_chat_model,
                    model=self.model_name,
                    prompt=s.messages_,
                    **self.spec_kwargs,
                )
                # Use a string for pattern matching.
                comp_for_match = comp[0] if isinstance(comp, list) else comp
                if self.spec_pattern_match(comp_for_match):
                    break

        for term in self.spec_format:
            s.text_ += term["text"]
            name = term["name"]
            if name is not None:
                s.variables[name] = term["text"]
                s.meta_info[name] = {}
                s.variable_event[name].set()

        self.spec_kwargs = {}
        self.spec_format = []

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if sampling_params.dtype is None:
            if self.is_chat_model:
                if not s.text_.endswith(self.chat_prefix):
                    raise RuntimeError(
                        "This use case is not supported. "
                        "For OpenAI chat models, sgl.gen must be right after sgl.assistant"
                    )
                prompt = s.messages_
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            generator = openai_completion_stream(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
            return generator
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        """Note: `choices_method` is not used by the OpenAI backend."""
        if self.is_chat_model:
            raise NotImplementedError(
                "select/choices is not supported for chat models. "
                "Please try to use a non-chat model such as gpt-3.5-turbo-instruct"
            )

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
            self.token_usage.prompt_tokens += ret.usage.prompt_tokens
            self.token_usage.completion_tokens = ret.usage.completion_tokens

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

        return ChoicesDecision(
            decision=choices[np.argmax(scores)],
            meta_info={"scores": scores},
        )


def openai_completion(
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs
) -> Union[str, List[str]]:
    # if "ebnf" is in kwargs, warn and remove
    if "ebnf" in kwargs:
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")
        del kwargs["ebnf"]

    for attempt in range(retries):
        try:
            if is_chat:
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                ret = client.chat.completions.create(messages=prompt, **kwargs)
                if len(ret.choices) == 1:
                    comp = ret.choices[0].message.content
                else:
                    comp = [c.message.content for c in ret.choices]
            else:
                ret = client.completions.create(prompt=prompt, **kwargs)
                if isinstance(prompt, (list, tuple)):
                    comp = [c.text for c in ret.choices]
                else:
                    comp = ret.choices[0].text
                    if len(ret.choices) > 1:
                        comp = [c.text for c in ret.choices]

            token_usage.prompt_tokens += ret.usage.prompt_tokens
            token_usage.completion_tokens += ret.usage.completion_tokens
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e

    return comp


def openai_completion_stream(
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs
):
    # if "ebnf" is in kwargs, warn and remove
    if "ebnf" in kwargs:
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")
        del kwargs["ebnf"]

    for attempt in range(retries):
        try:
            if is_chat:
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                generator = client.chat.completions.create(
                    messages=prompt,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                for ret in generator:
                    if len(ret.choices) == 0:
                        continue
                    try:
                        content = ret.choices[0].delta.content
                    except IndexError:
                        content = None
                    yield content or "", {}
            else:
                generator = client.completions.create(
                    prompt=prompt,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                for ret in generator:
                    if len(ret.choices) == 0:
                        continue
                    content = ret.choices[0].text
                    yield content or "", {}

            token_usage.prompt_tokens += ret.usage.prompt_tokens
            token_usage.completion_tokens += ret.usage.completion_tokens
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e
