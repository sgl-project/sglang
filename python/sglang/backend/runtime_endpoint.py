import json
from typing import Callable, List, Optional, Union

import numpy as np
import requests

from sglang.backend.base_backend import BaseBackend
from sglang.global_config import global_config
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglArgument, SglSamplingParams
from sglang.utils import encode_image_base64, find_printable_text, http_request


class RuntimeEndpoint(BaseBackend):
    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        verify: Optional[str] = None,
    ):
        super().__init__()
        self.support_concate_and_append = True

        self.base_url = base_url
        self.auth_token = auth_token
        self.api_key = api_key
        self.verify = verify

        res = http_request(
            self.base_url + "/get_model_info",
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        self.model_info = res.json()

        self.chat_template = get_chat_template_by_model_path(
            self.model_info["model_path"]
        )

    def get_model_name(self):
        return self.model_info["model_path"]

    def flush_cache(self):
        res = http_request(
            self.base_url + "/flush_cache",
            auth_token=self.auth_token,
            verify=self.verify,
        )
        self._assert_success(res)

    def get_server_args(self):
        res = http_request(
            self.base_url + "/get_server_args",
            auth_token=self.auth_token,
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
            auth_token=self.auth_token,
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
            auth_token=self.auth_token,
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
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if sampling_params.dtype is None:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                    **sampling_params.to_srt_kwargs(),
                },
            }
        elif sampling_params.dtype in [int, "int"]:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                    "dtype": "int",
                    **sampling_params.to_srt_kwargs(),
                },
            }
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            auth_token=self.auth_token,
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
        if sampling_params.dtype is None:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                    **sampling_params.to_srt_kwargs(),
                },
            }
        elif sampling_params.dtype in [int, "int"]:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                    "dtype": "int",
                    **sampling_params.to_srt_kwargs(),
                },
            }
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        data["stream"] = True
        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            stream=True,
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        pos = 0

        incomplete_text = ""
        for chunk in res.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                text = find_printable_text(data["text"][pos:])
                meta_info = data["meta_info"]
                pos += len(text)
                incomplete_text = data["text"][pos:]
                yield text, meta_info

        if len(incomplete_text) > 0:
            yield incomplete_text, meta_info

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
    ):
        assert temperature <= 1e-5

        # Cache common prefix
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        prompt_len = res.json()["meta_info"]["prompt_tokens"]

        # Compute logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": max(prompt_len - 2, 0),
        }
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        obj = res.json()
        normalized_prompt_logprobs = [
            r["meta_info"]["normalized_prompt_logprob"] for r in obj
        ]
        decision = choices[np.argmax(normalized_prompt_logprobs)]
        prefill_token_logprobs = [r["meta_info"]["prefill_token_logprobs"] for r in obj]
        decode_token_logprobs = [r["meta_info"]["decode_token_logprobs"] for r in obj]

        return (
            decision,
            normalized_prompt_logprobs,
            prefill_token_logprobs,
            decode_token_logprobs,
        )

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        res = http_request(
            self.base_url + "/concate_and_append_request",
            json={"src_rids": src_rids, "dst_rid": dst_rid},
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _add_images(self, s: StreamExecutor, data):
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]

    def _assert_success(self, res):
        if res.status_code != 200:
            raise RuntimeError(res.json())
