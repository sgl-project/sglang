import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import repeat
from typing import List, Optional, Union

from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams
from sglang.utils import http_request


class TGI(BaseBackend):
    def __init__(self, base_url):
        super().__init__()

        self.base_url = base_url

        res = http_request(self.base_url + "/info")
        assert res.status_code == 200
        self.model_info = res.json()
        self.chat_template = get_chat_template_by_model_path(
            self.model_info["model_id"]
        )

    def get_model_name(self):
        return self.model_info["model_id"]

    def get_chat_template(self):
        return self.chat_template

    @staticmethod
    def adapt_params(max_tokens, stop, sampling_params, **override_params):
        temperature = sampling_params.temperature
        do_sample = True
        if temperature == 0:
            do_sample = False
            temperature = None

        if stop is None:
            stop = []
        elif isinstance(stop, str):
            stop = [stop]

        top_p = sampling_params.top_p
        if top_p == 0:
            top_p = 0.001
        if top_p == 1:
            top_p = 0.999

        top_k = sampling_params.top_k
        if top_k == -1:
            top_k = None

        params = {
            "decoder_input_details": False,
            "details": False,
            "do_sample": do_sample,
            "max_new_tokens": max_tokens,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "return_full_text": False,
        }
        params.update(override_params)
        return params

    @staticmethod
    def _extract_int(text):
        words = re.split("\ |'|\/|\(|\)|\n|\.|,", text)
        for word in words:
            try:
                int(word)
                return word
            except ValueError:
                continue
        raise ValueError

    @staticmethod
    def _extract_choice(choices, text):
        # FIXME: Current only support the case where the choices are single words.
        words = re.split("\ |'|\/|\(|\)|\n|\.|,", text)
        for word in words:
            if word in choices:
                return word
        raise ValueError

    @staticmethod
    def _truncate_to_stop(text, stop):
        # The stop sequence may not be a single token. In this case TGI will generate
        # too many tokens so we need to truncate the output.
        if stop:
            stop = [stop] if isinstance(stop, str) else stop
            for stop_seq in stop:
                pos = text.find(stop_seq)
                if pos != -1:
                    return text[:pos]
        return text

    def _make_request(self, params):
        res = http_request(self.base_url + "/generate", json=params)
        if res.status_code != 200:
            raise ValueError(f"Error from TGI backend: {res.text}")
        return res.json()

    def retry_for_expected(self, prompt, params, extract_fn, retry=5):
        # TGI does not support logis_bias (yet), so we have to use an inefficient hack.
        failed = []
        while retry > 0:
            res_json = self._make_request(
                {
                    "inputs": prompt,
                    "parameters": params,
                }
            )
            text = res_json["generated_text"]
            try:
                return extract_fn(text)
            except ValueError:
                retry -= 1
            failed.append(text)

        msg = "=" * 20 + "\n"
        msg += f"Prompt:\n{prompt}\n"
        msg += "=" * 20 + "\n"
        for i, text in enumerate(failed):
            msg += f"====== Try {i+1}:\n{text}\n"

        raise ValueError(
            f"Model {self.model_info['model_id']} served by TGI backend does not generate"
            "expected output. Please improve the prompt, increase the temperature, or "
            f"use different models.\n{msg}"
        )

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        sampling_params: SglSamplingParams,
    ):
        decision = self.retry_for_expected(
            s.text_,
            self.adapt_params(16, [], sampling_params),
            partial(self._extract_choice, choices),
        )
        return decision, [1 if choice == decision else 0 for choice in choices]

    def generate(
        self,
        s: StreamExecutor,
        max_tokens: int,
        stop: Union[str, List[str]],
        sampling_params: SglSamplingParams,
        dtype: Optional[str] = None,
    ):
        if dtype is None:
            res_json = self._make_request(
                {
                    "inputs": s.text_,
                    "parameters": self.adapt_params(max_tokens, stop, sampling_params),
                }
            )
            return self._truncate_to_stop(res_json["generated_text"], stop), {}

        if dtype in [str, "str", "string"]:
            stop = ['"']
            res_json = self._make_request(
                {
                    "inputs": f'{s.text_}"',
                    "parameters": self.adapt_params(max_tokens, stop, sampling_params),
                }
            )
            return (
                '"' + self._truncate_to_stop(res_json["generated_text"], stop) + '"',
                {},
            )

        if dtype in [int, "int"]:
            return (
                self.retry_for_expected(
                    s.text_,
                    self.adapt_params(max_tokens, stop, sampling_params),
                    self._extract_int,
                ),
                {},
            )

        raise ValueError(f"Unknown dtype: {dtype}")
