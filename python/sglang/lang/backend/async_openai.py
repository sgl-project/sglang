"""Async OpenAI backend using httpx AsyncClient"""

import asyncio
import dataclasses
import logging
import warnings
from typing import AsyncIterator, List, Optional, Tuple, Union

import numpy as np

from sglang.lang.backend.async_base_backend import AsyncBaseBackend
from sglang.lang.chat_template import ChatTemplate, get_chat_template_by_model_path
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
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


@dataclasses.dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    def reset(self):
        self.prompt_tokens = self.completion_tokens = 0


class AsyncOpenAI(AsyncBaseBackend):
    """Async OpenAI backend using openai.AsyncOpenAI"""

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
            self.async_client = openai.AsyncAzureOpenAI(*args, **kwargs)
        else:
            self.async_client = openai.AsyncOpenAI(*args, **kwargs)

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
            self.is_chat_model = False

        self.chat_prefix = self.chat_template.role_prefix_and_suffix["assistant"][0]

        # Usage
        self.token_usage = TokenUsage(0, 0)

    def get_model_name(self):
        return self.model_name

    def get_chat_template(self):
        return self.chat_template

    async def generate(
        self,
        s,
        sampling_params: SglSamplingParams,
    ) -> Tuple[str, dict]:
        """Generate text asynchronously"""
        if sampling_params.dtype is None:
            if self.is_chat_model:
                if not s.text_.endswith(self.chat_prefix):
                    raise RuntimeError(
                        "For OpenAI chat models, sgl.gen must be right after sgl.assistant"
                    )
                prompt = s.messages_
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            comp = await async_openai_completion(
                client=self.async_client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
        elif sampling_params.dtype in [str, "str", "string"]:
            assert not self.is_chat_model, (
                "constrained type not supported on chat model"
            )
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = await async_openai_completion(
                client=self.async_client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_ + '"',
                stop=['"'],
                **kwargs,
            )
            if isinstance(comp, list):
                comp = ['"' + x + '"' for x in comp]
            else:
                comp = '"' + comp + '"'
        elif sampling_params.dtype in [int, "int"]:
            assert not self.is_chat_model, (
                "constrained type not supported on chat model"
            )
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = await async_openai_completion(
                client=self.async_client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_,
                logit_bias=self.logit_bias_int,
                stop=[" "],
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")

        return comp, {}

    async def generate_stream(
        self,
        s,
        sampling_params: SglSamplingParams,
    ) -> AsyncIterator[Tuple[str, dict]]:
        """Generate text asynchronously with streaming"""
        if sampling_params.dtype is None:
            if self.is_chat_model:
                if not s.text_.endswith(self.chat_prefix):
                    raise RuntimeError(
                        "For OpenAI chat models, sgl.gen must be right after sgl.assistant"
                    )
                prompt = s.messages_
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            async for chunk, meta_info in async_openai_completion_stream(
                client=self.async_client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            ):
                yield chunk, meta_info
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")

    async def select(
        self,
        s,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        """Select from choices - async version"""
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

            # Call API asynchronously
            ret = await self.async_client.completions.create(
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

    async def shutdown(self):
        """Shutdown the async client"""
        await self.async_client.close()


async def async_openai_completion(
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs
) -> Union[str, List[str]]:
    """Async version of openai_completion"""
    if "ebnf" in kwargs:
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")
        del kwargs["ebnf"]

    for attempt in range(retries):
        try:
            if is_chat:
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                ret = await client.chat.completions.create(messages=prompt, **kwargs)
                if len(ret.choices) == 1:
                    comp = ret.choices[0].message.content
                else:
                    comp = [c.message.content for c in ret.choices]
            else:
                ret = await client.completions.create(prompt=prompt, **kwargs)
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
            await asyncio.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e

    return comp


async def async_openai_completion_stream(
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs
) -> AsyncIterator[Tuple[str, dict]]:
    """Async version of openai_completion_stream"""
    if "ebnf" in kwargs:
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")
        del kwargs["ebnf"]

    for attempt in range(retries):
        try:
            if is_chat:
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                generator = await client.chat.completions.create(
                    messages=prompt,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                async for ret in generator:
                    if len(ret.choices) == 0:
                        continue
                    try:
                        content = ret.choices[0].delta.content
                    except IndexError:
                        content = None
                    yield content or "", {}
            else:
                generator = await client.completions.create(
                    prompt=prompt,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                async for ret in generator:
                    if len(ret.choices) == 0:
                        continue
                    content = ret.choices[0].text
                    yield content or "", {}

            # Get final usage stats
            if hasattr(ret, "usage") and ret.usage:
                token_usage.prompt_tokens += ret.usage.prompt_tokens
                token_usage.completion_tokens += ret.usage.completion_tokens
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")
            await asyncio.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e
