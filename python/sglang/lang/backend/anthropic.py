import uuid
from datetime import datetime
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

        debug_request_id = str(uuid.uuid4())
        s.log_debug(
            [
                {
                    "id": debug_request_id,
                    "requestPrompt": str(
                        [{"role": "system", "content": system}] + messages
                    ),
                    "requestTimestamp": datetime.now().isoformat(),
                    "requestMetadata": sampling_params.to_anthropic_kwargs(),
                }
            ]
        )

        ret = self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=messages,
            **sampling_params.to_anthropic_kwargs(),
        )
        comp = ret.content[0].text

        s.log_debug(
            [
                {
                    "id": debug_request_id,
                    "responseContent": comp,
                    "responseTimestamp": datetime.now().isoformat(),
                    "responseMetadata": ret.to_json(),
                }
            ]
        )

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

        debug_request_id = str(uuid.uuid4())
        debug_obj = s.log_debug(
            [
                {
                    "id": debug_request_id,
                    "requestPrompt": str(
                        [{"role": "system", "content": system}] + messages
                    ),
                    "requestTimestamp": datetime.now().isoformat(),
                    "requestMetadata": sampling_params.to_anthropic_kwargs(),
                }
            ]
        )

        with self.client.messages.stream(
            model=self.model_name,
            system=system,
            messages=messages,
            **sampling_params.to_anthropic_kwargs(),
        ) as stream:
            for text in stream.text_stream:
                yield text, {}
        final_message = stream.get_final_message()
        final_message_json = final_message.to_json()
        s.log_debug(
            [
                {
                    "id": debug_request_id,
                    "responseContent": final_message.content[0].text,
                    "responseTimestamp": datetime.now().isoformat(),
                    "responseMetadata": final_message_json,
                }
            ]
        )
