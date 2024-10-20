import json
import warnings
from typing import List, Optional

from sglang.global_config import global_config
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template, get_chat_template_by_model_path
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import (
    REGEX_BOOL,
    REGEX_FLOAT,
    REGEX_INT,
    REGEX_STR,
    SglSamplingParams,
)
from sglang.utils import http_request


class RuntimeEndpoint(BaseBackend):
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify: Optional[str] = None,
        chat_template_name: Optional[str] = None,
    ):
        super().__init__()
        self.support_concate_and_append = True

        self.base_url = base_url
        self.api_key = api_key
        self.verify = verify

        res = http_request(
            self.base_url + "/get_model_info",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        self.model_info = res.json()

        if chat_template_name:
            self.chat_template = get_chat_template(chat_template_name)
        else:
            self.chat_template = get_chat_template_by_model_path(
                self.model_info["model_path"]
            )

    def get_model_name(self):
        return self.model_info["model_path"]

    def flush_cache(self):
        res = http_request(
            self.base_url + "/flush_cache",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def get_server_args(self):
        res = http_request(
            self.base_url + "/get_server_args",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        return res.json()

    def get_chat_template(self):
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        res = http_request(
            self.base_url + "/generate",
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def commit_lazy_operations(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def fill_image(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _handle_dtype_to_regex(self, sampling_params: SglSamplingParams):
        if sampling_params.dtype is None:
            return

        if sampling_params.stop == ():
            sampling_params.stop = []

        dtype_regex = None
        if sampling_params.dtype in ["int", int]:

            dtype_regex = REGEX_INT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["float", float]:

            dtype_regex = REGEX_FLOAT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["str", str]:

            dtype_regex = REGEX_STR
        elif sampling_params.dtype in ["bool", bool]:

            dtype_regex = REGEX_BOOL
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        if dtype_regex is not None and sampling_params.regex is not None:
            warnings.warn(
                f"Both dtype and regex are set. Only dtype will be used. dtype: {sampling_params.dtype}, regex: {sampling_params.regex}"
            )

        sampling_params.regex = dtype_regex

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        self._handle_dtype_to_regex(sampling_params)
        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                **sampling_params.to_srt_kwargs(),
            },
        }

        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

        obj = res.json()
        comp = obj["text"]
        return comp, obj["meta_info"]

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        self._handle_dtype_to_regex(sampling_params)

        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                **sampling_params.to_srt_kwargs(),
            },
        }

        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        data["stream"] = True
        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            stream=True,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        pos = 0

        for chunk in res.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                chunk_text = data["text"][pos:]
                meta_info = data["meta_info"]
                pos += len(chunk_text)
                yield chunk_text, meta_info

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        assert temperature <= 1e-5

        # Cache common prefix
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        obj = self._generate_http_request(s, data)
        prompt_len = obj["meta_info"]["prompt_tokens"]
        logprob_start_len = max(prompt_len - 2, 0)  # For token healing

        # Compute logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
            },
            "return_logprob": True,
            "return_text_in_logprobs": True,
            "logprob_start_len": logprob_start_len,
        }
        obj = self._generate_http_request(s, data)

        normalized_prompt_logprobs = [
            r["meta_info"]["normalized_prompt_logprob"] for r in obj
        ]
        input_token_logprobs = [r["meta_info"]["input_token_logprobs"] for r in obj]
        output_token_logprobs = [r["meta_info"]["output_token_logprobs"] for r in obj]

        # Remove extra token if no token healing occurred
        for i in range(len(input_token_logprobs)):
            healed_token_str = input_token_logprobs[i][0][-1]
            if s.text_.endswith(healed_token_str):
                healed_token_logprob = input_token_logprobs[i][0][0]
                normalized_prompt_logprobs[i] = (
                    normalized_prompt_logprobs[i] * len(input_token_logprobs[i])
                    - healed_token_logprob
                ) / (len(input_token_logprobs[i]) - 1)
                input_token_logprobs[i] = input_token_logprobs[i][1:]

        # Compute unconditional logprobs if required
        if choices_method.requires_unconditional_logprobs:
            input_ids = [[el[1] for el in subl] for subl in input_token_logprobs]
            data = {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_logprob": True,
            }
            obj = self._generate_http_request(s, data)
            unconditional_token_logprobs = [
                r["meta_info"]["input_token_logprobs"] for r in obj
            ]
        else:
            unconditional_token_logprobs = None

        return choices_method(
            choices=choices,
            normalized_prompt_logprobs=normalized_prompt_logprobs,
            input_token_logprobs=input_token_logprobs,
            output_token_logprobs=output_token_logprobs,
            unconditional_token_logprobs=unconditional_token_logprobs,
        )

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        res = http_request(
            self.base_url + "/concate_and_append_request",
            json={"src_rids": src_rids, "dst_rid": dst_rid},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _generate_http_request(self, s: StreamExecutor, data):
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        return res.json()

    def _add_images(self, s: StreamExecutor, data):
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]

    def _assert_success(self, res):
        if res.status_code != 200:
            raise RuntimeError(res.json())
