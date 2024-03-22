from typing import List, Optional, Union

import numpy as np
from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

try:
    import anthropic
except ImportError as e:
    anthropic = e


class Anthropic(BaseBackend):
    def __init__(self, model_name):
        super().__init__()

        if isinstance(anthropic, Exception):
            raise anthropic

        self.model_name = model_name
        self.chat_template = get_chat_template("claude")

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

        ret = anthropic.Anthropic().messages.create(
            model=self.model_name,
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

        with anthropic.Anthropic().messages.stream(
            model=self.model_name,
            messages=messages,
            **sampling_params.to_anthropic_kwargs(),
        ) as stream:
            for text in stream.text_stream:
                yield text, {}
