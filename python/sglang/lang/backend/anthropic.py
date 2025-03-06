from typing import List, Optional, Union

import numpy as np

from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

try:
    import anthropic
except ImportError as e:
    anthropic = e


class Anthropic(BaseBackend):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()

        if isinstance(anthropic, Exception):
            raise anthropic

        self.model_name = model_name
        self.chat_template = get_chat_template("claude")
        self.client = anthropic.Anthropic(*args, **kwargs)

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

        if messages and messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
        else:
            system = ""

        ret = self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=messages,
            **sampling_params.to_anthropic_kwargs(),
        )
        comp = ret.content[0].text

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

        if messages and messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
        else:
            system = ""

        with self.client.messages.stream(
            model=self.model_name,
            system=system,
            messages=messages,
            **sampling_params.to_anthropic_kwargs(),
        ) as stream:
            for text in stream.text_stream:
                yield text, {}
