import json
from typing import Optional

from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams
from sglang.utils import http_request


class Shortfin(BaseBackend):
    def __init__(
        self,
        chat_template=None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__()

        if base_url is None:
            raise ValueError("`base_url` is required for Shortfin backend")

        self.chat_template = chat_template or get_chat_template_by_model_path(
            "llama-3-instruct"
        )

        self.client_params = {"base_url": base_url, "timeout": timeout}

    def _make_generate_request(self, shortfin_kwargs, stream=False):
        resp = http_request(
            f"{self.client_params['base_url']}/generate",
            json=shortfin_kwargs,
            timeout=self.client_params["timeout"],
            stream=stream,
        )
        self._assert_success(resp)
        return resp

    def _assert_success(self, res):
        if res.status_code != 200:
            raise RuntimeError(res.json())

    def _clean_response_message(self, text):
        return text[text.find(": ") + 2 :].rstrip("\n")

    def get_chat_template(self):
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        shortfin_kwargs = sampling_params.to_shortfin_kwargs()

        messages = s.text_
        shortfin_kwargs["text"] = messages

        resp = http_request(
            f"{self.client_params['base_url']}/generate",
            json=shortfin_kwargs,
            timeout=self.client_params["timeout"],
        )
        self._assert_success(resp)

        response_message = resp.resp.read().decode()
        response_message = self._clean_response_message(response_message)
        return response_message, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        shortfin_kwargs = sampling_params.to_shortfin_kwargs()
        shortfin_kwargs["stream"] = True

        messages = s.text_
        shortfin_kwargs["text"] = messages

        resp = http_request(
            f"{self.client_params['base_url']}/generate",
            json=shortfin_kwargs,
            stream=True,
            timeout=self.client_params["timeout"],
            method="POST",
        )
        self._assert_success(resp)
        for chunk in resp:
            if chunk == b"data: [DONE]\n\n":
                break
            text = chunk.decode()
            text = self._clean_response_message(text)
            if text is not None:
                yield text, {}
