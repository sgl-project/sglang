import logging
import time
import os

from sglang.backend.base_backend import BaseBackend

from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

import openai


logger = logging.getLogger("openrouter")


class OpenRouter(BaseBackend):
    def __init__(
        self,
        model_name: str,
        *args,
        **kwargs,
    ):
        super().__init__()

        if isinstance(openai, Exception):
            raise openai

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            *args,
            **kwargs,
        )

        self.model_name = model_name

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        kwargs = sampling_params.to_openrouter_kwargs()

        messages = s.messages_ or [{"role": "user", "content": s.text_}]

        comp = openrouter_completion(
            client=self.client,
            model=self.model_name,
            messages=messages,
            **kwargs,
        )

        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if sampling_params.dtype is None:
            kwargs = sampling_params.to_openrouter_kwargs()
            messages = s.messages_ or [{"role": "user", "content": s.text_}]
            generator = openrouter_completion_stream(
                client=self.client,
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            return generator
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")


def openrouter_completion(client, retries=3, messages=None, **kwargs):
    for attempt in range(retries):
        try:
            if "stop" in kwargs and kwargs["stop"] is None:
                kwargs.pop("stop")

            ret = client.chat.completions.create(messages=messages, **kwargs)
            comp = ret.choices[0].message.content

            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenRouter Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e

    return comp


def openrouter_completion_stream(client, retries=3, messages=None, **kwargs):
    for attempt in range(retries):
        try:
            if "stop" in kwargs and kwargs["stop"] is None:
                kwargs.pop("stop")
            generator = client.chat.completions.create(
                messages=messages, stream=True, **kwargs
            )
            for ret in generator:
                try:
                    content = ret.choices[0].delta.content
                except IndexError:
                    content = None
                yield content or "", {}
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            logger.error(f"OpenRouter Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e
