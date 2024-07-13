from typing import Mapping, Optional

from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

try:
    import litellm
except ImportError as e:
    litellm = e
    litellm.num_retries = 1


class LiteLLM(BaseBackend):
    def __init__(
        self,
        model_name,
        chat_template=None,
        api_key=None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 600,
        max_retries: Optional[int] = litellm.num_retries,
        default_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__()

        if isinstance(litellm, Exception):
            raise litellm

        self.model_name = model_name

        self.chat_template = chat_template or get_chat_template_by_model_path(
            model_name
        )

        self.client_params = {
            "api_key": api_key,
            "organization": organization,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
        }

    def get_chat_template(self):
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if s.messages_:
            messages = s.messages_
        else:
            messages = [{"role": "user", "content": s.text_}]

        ret = litellm.completion(
            model=self.model_name,
            messages=messages,
            **self.client_params,
            **sampling_params.to_anthropic_kwargs(),
        )
        comp = ret.choices[0].message.content

        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if s.messages_:
            messages = s.messages_
        else:
            messages = [{"role": "user", "content": s.text_}]

        ret = litellm.completion(
            model=self.model_name,
            messages=messages,
            stream=True,
            **self.client_params,
            **sampling_params.to_litellm_kwargs(),
        )
        for chunk in ret:
            text = chunk.choices[0].delta.content
            if text is not None:
                yield text, {}
