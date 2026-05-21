"""Async RuntimeEndpoint backend using aiohttp"""

import json
import warnings
from typing import AsyncIterator, Dict, List, Optional, Tuple

import aiohttp

from sglang.global_config import global_config
from sglang.lang.backend.async_base_backend import AsyncBaseBackend
from sglang.lang.chat_template import get_chat_template, get_chat_template_by_model_path
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.ir import (
    REGEX_BOOL,
    REGEX_FLOAT,
    REGEX_INT,
    REGEX_STR,
    SglSamplingParams,
)


class AsyncRuntimeEndpoint(AsyncBaseBackend):
    """Async version of RuntimeEndpoint using aiohttp"""

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

        self.model_info = None
        self.chat_template = None
        self._initialized = False
        self._chat_template_name = chat_template_name

    async def _ensure_initialized(self):
        """Lazy initialization - fetch model info on first use"""
        if not self._initialized:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                async with session.get(
                    f"{self.base_url}/get_model_info",
                    headers=headers,
                    ssl=self.verify if self.verify else None,
                ) as response:
                    await self._assert_success(response)
                    self.model_info = await response.json()

            if self._chat_template_name:
                self.chat_template = get_chat_template(self._chat_template_name)
            else:
                self.chat_template = get_chat_template_by_model_path(
                    self.model_info["model_path"]
                )

            self._initialized = True

    def get_model_name(self):
        if self.model_info is None:
            raise RuntimeError(
                "Backend not initialized. Call await backend._ensure_initialized() or run a query first."
            )
        return self.model_info["model_path"]

    async def flush_cache(self):
        await self._ensure_initialized()
        async with aiohttp.ClientSession() as session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.post(
                f"{self.base_url}/flush_cache",
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)

    async def get_server_info(self):
        await self._ensure_initialized()
        async with aiohttp.ClientSession() as session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.get(
                f"{self.base_url}/get_server_info",
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)
                return await response.json()

    def get_chat_template(self):
        if self.chat_template is None:
            raise RuntimeError(
                "Backend not initialized. Call await backend._ensure_initialized() or run a query first."
            )
        return self.chat_template

    async def cache_prefix(self, prefix_str: str):
        await self._ensure_initialized()
        data = {"text": prefix_str, "sampling_params": {"max_new_tokens": 0}}
        await self._generate_http_request(None, data)

    async def start_profile(self):
        await self._ensure_initialized()
        async with aiohttp.ClientSession() as session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.get(
                f"{self.base_url}/start_profile",
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)

    async def stop_profile(self):
        await self._ensure_initialized()
        async with aiohttp.ClientSession() as session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.get(
                f"{self.base_url}/stop_profile",
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)

    async def commit_lazy_operations(self, s):
        await self._ensure_initialized()
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        await self._generate_http_request(s, data)

    async def fill_image(self, s):
        await self._ensure_initialized()
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        await self._generate_http_request(s, data)

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

    async def generate(
        self,
        s,
        sampling_params: SglSamplingParams,
    ) -> Tuple[str, Dict]:
        await self._ensure_initialized()
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

        obj = await self._generate_http_request(s, data)
        comp = obj["text"]
        return comp, obj["meta_info"]

    async def generate_stream(
        self,
        s,
        sampling_params: SglSamplingParams,
    ) -> AsyncIterator[Tuple[str, Dict]]:
        await self._ensure_initialized()
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

        pos = 0
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.post(
                f"{self.base_url}/generate",
                json=data,
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line and line.startswith("data:"):
                        if line == "data: [DONE]":
                            break
                        chunk_data = json.loads(line[5:].strip())
                        chunk_text = chunk_data["text"][pos:]
                        meta_info = chunk_data["meta_info"]
                        pos += len(chunk_text)
                        yield chunk_text, meta_info

    async def select(
        self,
        s,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        await self._ensure_initialized()
        assert temperature <= 1e-5

        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        obj = await self._generate_http_request(s, data)
        prompt_len = obj["meta_info"]["prompt_tokens"]
        logprob_start_len = max(prompt_len - 2, 0)

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
        obj = await self._generate_http_request(s, data)

        input_token_logprobs = [r["meta_info"]["input_token_logprobs"] for r in obj]
        output_token_logprobs = [r["meta_info"]["output_token_logprobs"] for r in obj]
        normalized_prompt_logprobs = [
            compute_normalized_prompt_logprobs(r["meta_info"]["input_token_logprobs"])
            for r in obj
        ]

        for i in range(len(input_token_logprobs)):
            healed_token_str = input_token_logprobs[i][0][-1]
            if s.text_.endswith(healed_token_str):
                healed_token_logprob = input_token_logprobs[i][0][0]
                normalized_prompt_logprobs[i] = (
                    normalized_prompt_logprobs[i] * len(input_token_logprobs[i])
                    - healed_token_logprob
                ) / (len(input_token_logprobs[i]) - 1)
                input_token_logprobs[i] = input_token_logprobs[i][1:]

        if choices_method.requires_unconditional_logprobs:
            input_ids = [[el[1] for el in subl] for subl in input_token_logprobs]
            data = {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_logprob": True,
            }
            obj = await self._generate_http_request(s, data)
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

    async def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        await self._ensure_initialized()
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.post(
                f"{self.base_url}/concate_and_append_request",
                json={"src_rids": src_rids, "dst_rid": dst_rid},
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)

    async def _generate_http_request(self, s, data):
        if s is not None:
            self._add_images(s, data)

        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with session.post(
                f"{self.base_url}/generate",
                json=data,
                headers=headers,
                ssl=self.verify if self.verify else None,
            ) as response:
                await self._assert_success(response)
                return await response.json()

    def _add_images(self, s, data):
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]

    async def _assert_success(self, response):
        if response.status != 200:
            try:
                content = await response.json()
            except json.JSONDecodeError:
                content = await response.text()
            raise RuntimeError(content)


def compute_normalized_prompt_logprobs(input_logprobs):
    values = [x[0] for x in input_logprobs if x[0]]
    return sum(values) / len(values)
