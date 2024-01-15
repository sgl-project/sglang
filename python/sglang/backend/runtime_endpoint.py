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
    def __init__(self, base_url):
        super().__init__()
        self.support_concate_and_append = True

        self.base_url = base_url

        res = http_request(self.base_url + "/get_model_info")
        assert res.status_code == 200
        self.model_info = res.json()

        self.chat_template = get_chat_template_by_model_path(
            self.model_info["model_path"]
        )

    def get_model_name(self):
        return self.model_info["model_path"]

    def get_chat_template(self):
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        res = http_request(
            self.base_url + "/generate",
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},
        )
        assert res.status_code == 200

    def commit_lazy_operations(self, s: StreamExecutor):
        res = http_request(
            self.base_url + "/generate",
            json={"text": s.text_, "sampling_params": {"max_new_tokens": 0}},
        )
        assert res.status_code == 200

    def fill_image(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(self.base_url + "/generate", json=data)
        assert res.status_code == 200

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
                    **sampling_params.to_srt_kwargs(),
                },
            }
        elif sampling_params.dtype in [int, "int"]:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "dtype": "int",
                    **sampling_params.to_srt_kwargs(),
                },
            }
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        self._add_images(s, data)

        res = http_request(self.base_url + "/generate", json=data)
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
                    **sampling_params.to_srt_kwargs(),
                },
            }
        elif sampling_params.dtype in [int, "int"]:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "dtype": "int",
                    **sampling_params.to_srt_kwargs(),
                },
            }
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        data["stream"] = True
        self._add_images(s, data)

        response = http_request(self.base_url + "/generate", json=data, stream=True)
        pos = 0

        incomplete_text = ""
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
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
        res = http_request(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        prompt_len = res.json()["meta_info"]["prompt_tokens"]

        # Compute logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {"max_new_tokens": 0},
            "return_normalized_logprob": True,
            "normalized_logprob_start_len": prompt_len,
        }
        self._add_images(s, data)
        res = http_request(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        logps = [r["meta_info"]["normalized_logprob"] for r in res.json()]

        decision = choices[np.argmax(logps)]
        return decision, logps

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        res = http_request(
            self.base_url + "/concate_and_append_request",
            json={"src_rids": src_rids, "dst_rid": dst_rid},
        )
        assert res.status_code == 200

    def _add_images(self, s: StreamExecutor, data):
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]
