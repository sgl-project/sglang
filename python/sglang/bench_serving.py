# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

"""
Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
"""

import argparse
import asyncio
import importlib.util
import io
import json
import os
import pickle
import random
import resource
import shutil
import sys
import time
import traceback
import uuid
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pybase64
import requests
from datasets import load_dataset
from PIL import Image
from tqdm.asyncio import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

ASSISTANT_SUFFIX = "Assistant:"

TERM_PLOTLIB_AVAILABLE = (importlib.util.find_spec("termplotlib") is not None) and (
    shutil.which("gnuplot") is not None
)

global args


# don't want to import sglang package here
def _get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("true", "1")


def _create_bench_client_session():
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    # Define constants for timeout and buffer size for clarity and maintainability
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]
    timestamp: Optional[float] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    text_chunks: List[str] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    start_time: float = 0.0

    @staticmethod
    def init_new(request_func_input: RequestFuncInput):
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        return output


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def get_auth_headers() -> Dict[str, str]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return {"Authorization": f"Bearer {openai_api_key}"}
    else:
        api_key = os.environ.get("API_KEY")
        if api_key:
            return {"Authorization": f"{api_key}"}
        return {}


# trt llm does not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with _create_bench_client_session() as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.000001,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "min_length": request_func_input.output_len,
            "end_id": 1048576,
            **request_func_input.extra_request_body,
        }
        if args.disable_ignore_eos:
            del payload["min_length"]
            del payload["end_id"]
        output = RequestFuncOutput.init_new(request_func_input)

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input.prompt

    async with _create_bench_client_session() as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }

        # hack to accommodate different LoRA conventions between SGLang and vLLM.
        if request_func_input.lora_name:
            payload["model"] = request_func_input.lora_name
            payload["lora_path"] = request_func_input.lora_name

        if request_func_input.image_data:
            payload.update({"image_data": request_func_input.image_data})

        headers = get_auth_headers()

        output = RequestFuncOutput.init_new(request_func_input)

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.text_chunks.append(
                                        data["choices"][0]["text"]
                                    )
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                                output_len = (data.get("usage") or {}).get(
                                    "completion_tokens", output_len
                                )

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Makes a request to the OpenAI Chat Completions API.

    Handles both streaming and non-streaming responses, including support
    for image data in messages. Calculates and returns various performance
    metrics.

    Args:
        request_func_input: Input parameters for the request.
        pbar: Optional tqdm progress bar to update.

    Returns:
        RequestFuncOutput: Output of the request, including generated text,
                           latency, TTFT, ITL, and success status.
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    # TODO put it to other functions when `pbar` logic is refactored
    if getattr(args, "print_requests", False):
        rid = str(uuid.uuid4())
        input_partial = deepcopy(request_func_input)
        input_partial.prompt = "..."
        request_start_time = time.time()
        print(
            f'rid={rid} time={request_start_time} message="request start" request_func_input="{str(input_partial)}"'
        )

    if request_func_input.image_data:
        # Build multi-image content: a list of image_url entries followed by the text
        content_items = [
            {
                "type": "image_url",
                "image_url": {"url": img_url},
            }
            for img_url in request_func_input.image_data
        ]
        content_items.append({"type": "text", "text": request_func_input.prompt})
        messages = [
            {
                "role": "user",
                "content": content_items,
            },
        ]
    else:
        messages = [{"role": "user", "content": request_func_input.prompt}]

    async with _create_bench_client_session() as session:
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }

        # hack to accommodate different LoRA conventions between SGLang and vLLM.
        if request_func_input.lora_name:
            payload["model"] = request_func_input.lora_name
            payload["lora_path"] = request_func_input.lora_name

        headers = get_auth_headers()

        output = RequestFuncOutput.init_new(request_func_input)

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    if args.disable_stream:
                        # Non-streaming response
                        response_json = await response.json()
                        output.generated_text = response_json["choices"][0]["message"][
                            "content"
                        ]
                        output.success = True
                        output.latency = time.perf_counter() - st
                        output.ttft = (
                            output.latency
                        )  # For non-streaming, TTFT = total latency
                        output.output_len = response_json.get("usage", {}).get(
                            "completion_tokens", output_len
                        )
                    else:
                        # Streaming response
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                            latency = time.perf_counter() - st
                            if chunk == "[DONE]":
                                pass
                            else:
                                data = json.loads(chunk)

                                # Check if this chunk contains content
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.text_chunks.append(content)
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    most_recent_timestamp = timestamp
                                    generated_text += content

                                # Check for usage info in final chunk
                                output_len = (data.get("usage") or {}).get(
                                    "completion_tokens", output_len
                                )

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    # TODO put it to other functions when `pbar` logic is refactored
    if getattr(args, "print_requests", False):
        curr_t = time.time()
        output_partial = deepcopy(output)
        output_partial.generated_text = "..."
        print(
            f'rid={rid} time={curr_t} time_delta={curr_t - request_start_time} message="request end" output="{str(output_partial)}"'
        )

    if pbar:
        pbar.update(1)
    return output


async def async_request_truss(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    prompt = request_func_input.prompt

    async with _create_bench_client_session() as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput.init_new(request_func_input)

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_sglang_generate(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    prompt = request_func_input.prompt

    async with _create_bench_client_session() as session:
        payload = {
            ("text" if isinstance(prompt, str) else "input_ids"): prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": request_func_input.output_len,
                "ignore_eos": not args.disable_ignore_eos,
            },
            "stream": not args.disable_stream,
            "lora_path": request_func_input.lora_name,
            "return_logprob": args.return_logprob,
            "logprob_start_len": -1,
            **request_func_input.extra_request_body,
        }

        # Add image data if available (list of image urls/base64)
        if request_func_input.image_data:
            payload["image_data"] = request_func_input.image_data

        headers = get_auth_headers()

        output = RequestFuncOutput.init_new(request_func_input)

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        last_output_len = 0
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if "text" in data and data["text"]:
                                timestamp = time.perf_counter()
                                generated_text = data["text"]
                                output_len = data["meta_info"]["completion_tokens"]

                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    num_new_tokens = output_len - last_output_len
                                    if num_new_tokens == 0:
                                        continue
                                    chunk_gap = timestamp - most_recent_timestamp
                                    adjust_itl = chunk_gap / num_new_tokens
                                    output.itl.extend([adjust_itl] * num_new_tokens)

                                most_recent_timestamp = timestamp
                                last_output_len = output_len

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(f"{output.error=}")

    if pbar:
        pbar.update(1)
    return output


async def async_request_gserver(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError()


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with _create_bench_client_session() as session:
        output = RequestFuncOutput()
        try:
            body = {
                "activities": getattr(args, "profile_activities", []),
                "num_steps": getattr(args, "profile_num_steps", None),
                "profile_by_stage": getattr(args, "profile_by_stage", None),
                "profile_stages": getattr(args, "profile_stages", None),
            }
            print(f"async_request_profile {api_url=} {body=}")
            async with session.post(url=api_url, json=body) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def _build_profile_urls(
    profile_prefill_url: Optional[List[str]],
    profile_decode_url: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """Build profile URLs list from prefill/decode URL arguments.

    Returns:
        List of (worker_type, url) tuples. e.g., [("Prefill-0", "http://..."), ("Decode-0", "http://...")]
    """
    profile_urls = []
    if profile_prefill_url:
        for idx, url in enumerate(profile_prefill_url):
            profile_urls.append((f"Prefill-{idx}", url))
    if profile_decode_url:
        for idx, url in enumerate(profile_decode_url):
            profile_urls.append((f"Decode-{idx}", url))
    return profile_urls


async def _call_profile_pd(profile_urls: List[Tuple[str, str]], mode: str) -> None:
    """Call profile endpoint (start/stop) on PD separated workers.

    Args:
        profile_urls: List of (worker_type, url) tuples
        mode: "start" or "stop"
    """
    endpoint = "/start_profile" if mode == "start" else "/stop_profile"
    action = "Starting" if mode == "start" else "Stopping"
    action_past = "started" if mode == "start" else "stopped"

    print(f"{action} profiler...")

    for worker_type, url in profile_urls:
        profile_output = await async_request_profile(api_url=url + endpoint)
        if profile_output.success:
            print(f"Profiler {action_past} for {worker_type} worker at {url}")
        else:
            print(
                f"Failed to {mode} profiler for {worker_type} worker at {url}: {profile_output.error}"
            )


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() == "true":
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    assert (
        pretrained_model_name_or_path is not None
        and pretrained_model_name_or_path != ""
    )
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def get_processor(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    assert (
        pretrained_model_name_or_path is not None
        and pretrained_model_name_or_path != ""
    )
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.utils.hf_transformers_utils import get_processor

        return get_processor(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def get_dataset(args, tokenizer, model_id=None):
    tokenize_prompt = getattr(args, "tokenize_prompt", False)
    if args.dataset_name == "sharegpt":
        assert not tokenize_prompt
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )
    elif args.dataset_name.startswith("random"):
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            random_sample=args.dataset_name == "random",
            return_text=not tokenize_prompt,
        )
    elif args.dataset_name == "image":
        processor = get_processor(model_id)
        input_requests = sample_image_requests(
            num_requests=args.num_prompts,
            image_count=args.image_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            processor=processor,
            image_content=args.image_content,
            image_format=args.image_format,
            image_resolution=args.image_resolution,
            backend=args.backend,
        )
    elif args.dataset_name == "generated-shared-prefix":
        assert not tokenize_prompt
        input_requests = sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            tokenizer=tokenizer,
            args=args,
        )
    elif args.dataset_name == "mmmu":
        processor = get_processor(model_id)
        input_requests = sample_mmmu_requests(
            num_requests=args.num_prompts,
            processor=processor,
            backend=args.backend,
            fixed_output_len=args.random_output_len,
            random_sample=True,
        )
    elif args.dataset_name == "mooncake":
        # For mooncake, we don't generate the prompts here.
        # We just load the raw trace data. The async generator will handle the rest.
        if not args.dataset_path:
            local_path = os.path.join("/tmp", args.mooncake_workload + "_trace.jsonl")
        else:
            local_path = args.dataset_path

        if not os.path.exists(local_path):
            download_and_cache_file(
                MOONCAKE_DATASET_URL[args.mooncake_workload], local_path
            )

        with open(local_path, "r") as f:
            all_requests_data = [json.loads(line) for line in f if line.strip()]

        # Limit the number of requests based on --num-prompts
        input_requests = all_requests_data[: args.num_prompts]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return input_requests


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_sglang_generate,
    "sglang-native": async_request_sglang_generate,
    "sglang-oai": async_request_openai_completions,
    "sglang-oai-chat": async_request_openai_chat_completions,
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
    "lmdeploy": async_request_openai_completions,
    "lmdeploy-chat": async_request_openai_chat_completions,
    "trt": async_request_trt_llm,
    "gserver": async_request_gserver,
    "truss": async_request_truss,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_input_text: int
    total_input_vision: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    max_output_tokens_per_s: float = 0.0
    max_concurrent_requests: int = 0


SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
MOONCAKE_DATASET_URL = {
    "mooncake": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    "conversation": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    "synthetic": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    "toolagent": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
}


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if is_file_valid_json(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_file_valid_json(path):
    if not os.path.isfile(path):
        return False

    # TODO can fuse into the real file open later
    try:
        with open(path) as f:
            json.load(f)
        return True
    except JSONDecodeError as e:
        print(
            f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
        )
        return False


@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0


async def get_mooncake_request_over_time(
    input_requests: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    slowdown_factor: float,
    num_rounds: int,
) -> AsyncGenerator[DatasetRow, None]:
    """
    An async generator that yields requests based on the timestamps in the Mooncake trace file,
    with support for multi-round sessions.
    """
    if not input_requests:
        return

    input_requests.sort(key=lambda r: r["timestamp"])

    start_time = time.perf_counter()
    trace_start_time_ms = input_requests[0]["timestamp"]

    for record in input_requests:
        # Calculate when this entire session should start
        relative_arrival_time_s = (record["timestamp"] - trace_start_time_ms) / 1000.0
        target_arrival_time_s = relative_arrival_time_s * slowdown_factor

        current_elapsed_time_s = time.perf_counter() - start_time
        sleep_duration_s = target_arrival_time_s - current_elapsed_time_s
        if sleep_duration_s > 0:
            await asyncio.sleep(sleep_duration_s)

        # Once the session starts, generate all rounds for it as a burst
        # This simulates a user engaging in a multi-turn conversation

        # Base user query constructed from hash_ids
        user_query_base = ""
        hash_ids = record.get("hash_ids", [])
        for hash_id in hash_ids:
            user_query_base += f"{hash_id}" + " ".join(
                ["hi"] * 128
            )  # Shorter for multi-round
        user_query_base += "Tell me a story based on this context."

        output_len_per_round = record.get("output_length", 256)
        chat_history = []

        for i in range(num_rounds):
            # Add user query for the current round
            chat_history.append(
                {"role": "user", "content": f"Round {i + 1}: {user_query_base}"}
            )

            # Form the full prompt from history
            try:
                full_prompt_text = tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            except Exception:
                full_prompt_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in chat_history]
                )

            prompt_len = len(tokenizer.encode(full_prompt_text))

            yield DatasetRow(
                prompt=full_prompt_text,
                prompt_len=prompt_len,
                output_len=output_len_per_round,
            )

            # Add a placeholder assistant response for the next round's context
            # We use a placeholder because we don't know the real response
            placeholder_response = " ".join(["story"] * output_len_per_round)
            chat_history.append({"role": "assistant", "content": placeholder_response})


def sample_mmmu_requests(
    num_requests: int,
    processor: AutoProcessor | AutoTokenizer,
    backend: str = "sglang",
    fixed_output_len: Optional[int] = None,
    random_sample: bool = True,
) -> List[DatasetRow]:
    """
    Sample requests from the MMMU dataset using HuggingFace datasets.

    Args:
        num_requests: Number of requests to sample.
        fixed_output_len: If provided, use this fixed output length for all requests.
        random_sample: Whether to randomly sample or take the first N.

    Returns:
        List of tuples (prompt, prompt_token_len, output_token_len).
    """
    print("Loading MMMU dataset from HuggingFace...")

    try:
        print("Attempting to load MMMU Math dataset...")
        mmmu_dataset = load_dataset("MMMU/MMMU", "Math", split="test")
        print(
            f"Successfully loaded MMMU Math dataset from HuggingFace with {len(mmmu_dataset)} examples"
        )
    except Exception as e:
        print(f"Failed to load MMMU Math dataset: {e}")
        raise ValueError(f"Failed to load MMMU dataset: {e}")

    # Sample from the dataset
    if len(mmmu_dataset) > num_requests:
        if random_sample:
            # Random sample
            indices = random.sample(range(len(mmmu_dataset)), num_requests)
            sample_dataset = mmmu_dataset.select(indices)
        else:
            # Take first N
            sample_dataset = mmmu_dataset.select(
                range(min(num_requests, len(mmmu_dataset)))
            )
    else:
        print(f"Dataset has less than {num_requests} examples, using all examples")
        sample_dataset = mmmu_dataset

    print(f"Selected {len(sample_dataset)} examples for benchmarking")

    # Create prompts
    filtered_dataset = []

    for i, example in enumerate(sample_dataset):
        try:
            # Extract image_1
            image = example.get("image_1")

            if image is not None:
                if hasattr(image, "save"):
                    # Convert RGBA images to RGB before encoding
                    if image.mode == "RGBA":
                        image = image.convert("RGB")

                    # Encode image to base64 (save as PNG to support palette/alpha modes)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = pybase64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_data = f"data:image/png;base64,{img_str}"
                else:
                    continue

                # Extract the question
                question = example.get("question")

                # Construct the prompt
                text_prompt = f"Question: {question}\n\nAnswer: "
                output_len = fixed_output_len if fixed_output_len is not None else 256
                data_row = create_mm_data_row(
                    text_prompt, [image], [image_data], output_len, processor, backend
                )
                filtered_dataset.append(data_row)

        except Exception as e:
            print(f"Error processing example {i}: {e}")

    print(f"\nCreated {len(filtered_dataset)} MMMU prompts")
    return filtered_dataset


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not is_file_valid_json(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset


def compute_random_lens(full_len: int, range_ratio: float, num: int):
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    )


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
) -> List[DatasetRow]:
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )

    if random_sample:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not is_file_valid_json(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[DatasetRow] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            input_content = input_ids
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=int(input_lens[i]),
                    output_len=int(output_lens[i]),
                )
            )
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            input_content = [
                (offsets[i] + i + j) % tokenizer.vocab_size
                for j in range(input_lens[i])
            ]
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=int(input_lens[i]),
                    output_len=int(output_lens[i]),
                )
            )

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


def parse_image_resolution(image_resolution: str) -> Tuple[int, int]:
    """Parse image resolution into (width, height).

    Supports presets '1080p', '720p', '360p' and custom 'heightxwidth' format
    (e.g., '1080x1920' means height=1080, width=1920).
    """
    resolution_to_size = {
        "4k": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "360p": (640, 360),
    }
    if image_resolution in resolution_to_size:
        return resolution_to_size[image_resolution]

    res = image_resolution.strip().lower()
    if "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            height = int(parts[0])
            width = int(parts[1])
            if height > 0 and width > 0:
                return (width, height)

    raise ValueError(
        f"Unsupported image resolution: {image_resolution}. "
        "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920)."
    )


def create_mm_data_row(
    text_prompt, images: list, images_base64, output_len, processor, backend
):
    try:
        if type(processor).__name__ == "Phi4MMProcessor":
            # <|endoftext10|> is the image token used in the phi-4-multimodal model.
            content_items = text_prompt.replace("image 1", "|endoftext10|")
        else:
            content_items = [
                {"type": "image", "image": {"url": image_base64}}
                for image_base64 in images_base64
            ]
            content_items.append({"type": "text", "text": text_prompt})
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        # Note (Xinyuan): This is a workaround for an issue where some tokenizers do not support content as a list. (e.g. InternVL)
        print(f"Error applying chat template: {e}, fallback to <image> tag")
        # Some tokenizers do not support list content; fall back to a placeholder in the text
        prompt_str = f"<image>{text_prompt}"

    # Calculate total tokens (text + vision)
    prompt_len = processor(
        text=[prompt_str],
        images=images,
        padding=False,
        return_tensors="pt",
    )["input_ids"].numel()

    # Calculate text-only tokens
    try:
        # Create text-only version of the prompt
        text_only_prompt = processor.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_prompt],
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()
    except Exception:
        # Fallback: just tokenize the text prompt directly
        tokenizer_to_use = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        text_prompt_len = len(tokenizer_to_use.encode(text_prompt))

    # Vision tokens = total tokens - text tokens
    vision_prompt_len = prompt_len - text_prompt_len

    use_raw_prompt = backend in [
        "sglang",
        "sglang-oai",
        "sglang-oai-chat",
        "vllm",
        "vllm-chat",
        "lmdeploy",
        "lmdeploy-chat",
    ]
    return DatasetRow(
        prompt=text_prompt if use_raw_prompt else prompt_str,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
    )


def sample_image_requests(
    num_requests: int,
    image_count: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    processor: AutoProcessor,
    image_content: str,
    image_format: str,
    image_resolution: str,
    backend: str,
) -> List[DatasetRow]:
    """Generate requests with images.

    - Each request includes ``image_count`` images.
    - Supported resolutions: 4k (3840x2160), 1080p (1920x1080), 720p (1280x720), 360p (640x360),
      or custom 'heightxwidth' (e.g., 1080x1920).
    - Text lengths follow the 'random' dataset sampling rule. ``prompt_len``
      only counts text tokens and excludes image data.
    """

    # Parse resolution (supports presets and 'heightxwidth')
    width, height = parse_image_resolution(image_resolution)

    # Check for potentially problematic combinations and warn user
    if width * height >= 1920 * 1080 and image_count * num_requests >= 100:
        warnings.warn(
            f"High resolution ({width}x{height}) with {image_count * num_requests} total images "
            f"may take a long time. Consider reducing resolution or image count.",
            UserWarning,
            stacklevel=2,
        )

    # Sample text lengths
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_requests,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_requests,
    )

    def _gen_random_image_data_uri(
        width: int = width, height: int = height
    ) -> (Image, str, int):
        if image_content == "blank":
            # Generate blank white image
            arr = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            # Generate random colored image
            arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format=image_format, quality=85)
        encoded = pybase64.b64encode(buf.getvalue()).decode("utf-8")
        image_data = f"data:image/{image_format};base64,{encoded}"
        image_bytes = len(image_data.encode("utf-8"))
        return img, image_data, image_bytes

    dataset: List[DatasetRow] = []
    total_image_bytes = 0
    for i in range(num_requests):
        # Generate text prompt
        text_prompt = gen_mm_prompt(
            processor.tokenizer,
            processor.image_token_id if hasattr(processor, "image_token_id") else None,
            int(input_lens[i]),
        )

        # Generate image list
        images, images_base64, images_bytes = zip(
            *[_gen_random_image_data_uri() for _ in range(image_count)]
        )
        total_image_bytes += sum(list(images_bytes))

        data_row = create_mm_data_row(
            text_prompt,
            list(images),
            list(images_base64),
            int(output_lens[i]),
            processor,
            backend,
        )

        dataset.append(data_row)

    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    print(
        f"\nCreated {len(dataset)} {image_content} {image_format} images with average {total_image_bytes // num_requests} bytes per request"
    )
    return dataset


@lru_cache(maxsize=1)
def get_available_tokens(tokenizer):
    """Get all available token ids from the tokenizer vocabulary."""
    return list(tokenizer.get_vocab().values())


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    if image_pad_id:
        all_available_tokens.remove(image_pad_id)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def get_gen_prefix_cache_path(args, tokenizer):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    # Create a unique cache filename based on the generation parameters
    cache_key = (
        f"gen_shared_prefix_{args.seed}_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
        f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_{args.gsp_output_len}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> List[DatasetRow]:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(args, tokenizer)

    # Try to load from cache first
    if cache_path.exists() and range_ratio == 1:
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(
        f"\nGenerating new input data... "
        f"({num_groups=}, {prompts_per_group}, {system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=})"
    )

    system_prompt_lens = compute_random_lens(
        full_len=system_prompt_len,
        range_ratio=range_ratio,
        num=num_groups,
    )
    question_lens = compute_random_lens(
        full_len=question_len,
        range_ratio=range_ratio,
        num=num_groups * prompts_per_group,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_groups * prompts_per_group,
    )
    del system_prompt_len, question_len, output_len

    # Generate system prompts for each group
    system_prompts = []
    for i in range(num_groups):
        system_prompt = gen_prompt(tokenizer, system_prompt_lens[i].item())
        system_prompts.append(system_prompt)

    # Generate questions
    questions = []
    for i in range(num_groups * prompts_per_group):
        question = gen_prompt(tokenizer, question_lens[i].item())
        questions.append(question)

    # Combine system prompts with questions
    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0

    for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
        system_prompt = system_prompts[group_idx]
        for prompt_idx in tqdm(
            range(prompts_per_group), desc="Generating questions", leave=False
        ):
            flat_index = group_idx * prompts_per_group + prompt_idx
            question = questions[flat_index]
            full_prompt = f"{system_prompt}\n\n{question}"
            prompt_len = len(tokenizer.encode(full_prompt))

            input_requests.append(
                DatasetRow(
                    prompt=full_prompt,
                    prompt_len=prompt_len,
                    output_len=output_lens[flat_index].item(),
                )
            )
            total_input_tokens += prompt_len
            total_output_tokens += output_lens[flat_index].item()

    # Shuffle questions
    random.shuffle(input_requests)

    # Print statistics
    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Total prompts: {len(input_requests)}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(
        f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
    )
    print(
        f"Average question length: {sum(len(tokenizer.encode(q)) for q in questions) / len(questions):.1f} tokens\n"
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching generated input data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(input_requests, f)

    return input_requests


async def get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
) -> AsyncGenerator[DatasetRow, None]:
    if use_trace_timestamps:
        print(
            f"Using trace timestamps for request generation with slowdown factor {slowdown_factor}."
        )
        # Sort requests by timestamp for correct replay
        input_requests.sort(key=lambda r: r.timestamp)

        start_time = time.perf_counter()
        trace_start_time_ms = input_requests[0].timestamp if input_requests else 0

        for request in input_requests:
            trace_time_s = (request.timestamp - trace_start_time_ms) / 1000.0
            target_arrival_time = start_time + (trace_time_s * slowdown_factor)

            sleep_duration = target_arrival_time - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            yield request
    else:
        input_requests_iter = iter(input_requests)
        for request in input_requests_iter:
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[DatasetRow],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
    accept_length: Optional[float] = None,
    plot_throughput: bool = False,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    total_input_text = 0
    total_input_vision = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    retokenized_itls: List[float] = []

    use_retokenized_itl = (
        accept_length is not None
        and accept_length > 0
        and backend in ("sglang-oai", "sglang-oai-chat")
    )

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i].prompt_len
            total_input_text += input_requests[i].text_prompt_len
            total_input_vision += input_requests[i].vision_prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            if use_retokenized_itl:
                for k, itl in enumerate(outputs[i].itl):
                    num_tokens = len(
                        tokenizer.encode(
                            outputs[i].text_chunks[k], add_special_tokens=False
                        )
                    )
                    adjusted_itl = itl / num_tokens
                    retokenized_itls.extend([adjusted_itl] * num_tokens)
            else:
                itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    successful_outputs = [output for output in outputs if output.success]
    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(
            output.start_time + output.latency for output in successful_outputs
        )

        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for output in outputs:
            if not output.success:
                continue

            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int(
                (output.start_time + output.latency) - min_start_time
            )

            for second in range(
                request_start_second, min(request_end_second + 1, duration_seconds)
            ):
                concurrent_requests_per_second[second] += 1

        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

        if plot_throughput:
            if TERM_PLOTLIB_AVAILABLE:
                import termplotlib as tpl

                fig = tpl.figure()
                fig.plot(
                    np.arange(len(tokens_per_second)),
                    tokens_per_second,
                    title="Output tokens per second",
                    xlabel="Time (s)",
                )
                fig.plot(
                    np.arange(len(concurrent_requests_per_second)),
                    concurrent_requests_per_second,
                    title="Concurrent requests per second",
                    xlabel="Time (s)",
                )
                fig.show()
            else:
                print("tip: install termplotlib and gnuplot to plot the metrics")

    itls = retokenized_itls if use_retokenized_itl else itls
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_input_text=total_input_text,
        total_input_vision=total_input_vision,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[DatasetRow],
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    lora_names: List[str],
    lora_request_distribution: Optional[str],
    lora_zipf_alpha: Optional[float],
    extra_request_body: Dict[str, Any],
    profile: bool,
    pd_separated: bool = False,
    flush_cache: bool = False,
    warmup_requests: int = 1,
    use_trace_timestamps: bool = False,
    mooncake_slowdown_factor=1.0,
    mooncake_num_rounds=1,
    profile_prefill_url: Optional[List[str]] = None,
    profile_decode_url: Optional[List[str]] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    # Warmup
    print(f"Starting warmup with {warmup_requests} sequences...")

    # Handle the data structure difference for the warmup request
    if args.dataset_name == "mooncake":
        # For mooncake, input_requests is a list of dicts.
        # We need to build a temporary DatasetRow for the warmup phase.
        warmup_record = input_requests[0]

        # Build prompt from hash_ids, just like in the async generator
        hash_ids = warmup_record.get("hash_ids", [])
        prompt_text = ""
        for hash_id in hash_ids:
            prompt_text += f"{hash_id}" + " ".join(["hi"] * 512)
        prompt_text += "Can you tell me a detailed story in 1000 words?"

        output_len = warmup_record.get("output_length", 32)
        prompt_len = len(tokenizer.encode(prompt_text))

        # Create a temporary DatasetRow object for warmup
        test_request = DatasetRow(
            prompt=prompt_text,
            prompt_len=prompt_len,
            output_len=output_len,
            image_data=None,  # Mooncake doesn't have image data
        )
    else:
        # For all other datasets, input_requests is a list of DatasetRow objects
        test_request = input_requests[0]

    if lora_names is not None and len(lora_names) != 0:
        lora_name = lora_names[0]
    else:
        lora_name = None

    # Create the test input once
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=min(test_request.output_len, 32),
        lora_name=lora_name,
        image_data=test_request.image_data,
        extra_request_body=extra_request_body,
    )

    # Run warmup requests
    warmup_tasks = []
    for _ in range(warmup_requests):
        warmup_tasks.append(
            asyncio.create_task(request_func(request_func_input=test_input))
        )

    warmup_outputs = await asyncio.gather(*warmup_tasks)

    # Check if at least one warmup request succeeded
    if warmup_requests > 0 and not any(output.success for output in warmup_outputs):
        raise ValueError(
            "Warmup failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {warmup_outputs[0].error}"
        )
    else:
        print(
            f"Warmup completed with {args.warmup_requests} sequences. Starting main benchmark run..."
        )

    # Flush cache
    if ("sglang" in backend and _get_bool_env_var("SGLANG_IS_IN_CI")) or flush_cache:
        requests.post(base_url + "/flush_cache", headers=get_auth_headers())

    time.sleep(1.0)

    # Build profile URLs for PD separated mode (do this once at the beginning)
    pd_profile_urls = []
    if profile and pd_separated:
        pd_profile_urls = _build_profile_urls(profile_prefill_url, profile_decode_url)
        if not pd_profile_urls:
            print(
                "Warning: PD separated mode requires --profile-prefill-url or --profile-decode-url"
            )
            print("Skipping profiler start. Please specify worker URLs for profiling.")

    # Start profiler
    if profile:
        if pd_separated:
            if pd_profile_urls:
                await _call_profile_pd(pd_profile_urls, "start")
        else:
            print("Starting profiler...")
            profile_output = await async_request_profile(
                api_url=base_url + "/start_profile"
            )
            if profile_output.success:
                print("Profiler started")

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    pbar_total = len(input_requests)
    if (
        backend == "sglang" and args.dataset_name == "mooncake"
    ):  # Assuming mooncake is mainly for sglang or similar backends
        print("Using time-based Mooncake request scheduler, ignoring --request-rate.")
        request_generator = get_mooncake_request_over_time(
            input_requests, tokenizer, mooncake_slowdown_factor, mooncake_num_rounds
        )
        print(
            f"Starting Mooncake trace replay. Sessions: {len(input_requests)}, Rounds per session: {mooncake_num_rounds}. Slowdown factor: {mooncake_slowdown_factor}"
        )
        pbar_total *= args.mooncake_num_rounds
    else:
        request_generator = get_request(input_requests, request_rate)

    # Prepare LoRA request distribution parameters
    if lora_request_distribution == "distinct":
        lora_idx = 0
    elif lora_request_distribution == "skewed":
        weights = np.array([lora_zipf_alpha**-i for i in range(len(lora_names))])
        lora_probs = weights / np.sum(weights)
    else:
        lora_idx = None
        lora_probs = None

    pbar = None if disable_tqdm else tqdm(total=pbar_total)
    async for request in request_generator:
        if lora_names is not None and len(lora_names) != 0:
            if lora_request_distribution == "uniform":
                lora_name = random.choice(lora_names)
            elif lora_request_distribution == "distinct":
                lora_name = lora_names[lora_idx]
                lora_idx = (lora_idx + 1) % len(lora_names)
            else:
                assert (
                    lora_request_distribution == "skewed"
                ), f"Unexpected lora_request_distribution: {lora_request_distribution}. Expected 'skewed'."

                lora_name = np.random.choice(lora_names, p=lora_probs)
        else:
            lora_name = None

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
            lora_name=lora_name,
            image_data=request.image_data,
            extra_request_body=extra_request_body,
            timestamp=request.timestamp,
        )

        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    # Stop profiler
    if profile:
        if pd_separated:
            if pd_profile_urls:
                await _call_profile_pd(pd_profile_urls, "stop")
        else:
            if getattr(args, "profile_num_steps", None) is None:
                print("Stopping profiler...")
                profile_output = await async_request_profile(
                    api_url=base_url + "/stop_profile"
                )
                if profile_output.success:
                    print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    if "sglang" in backend:
        server_info = requests.get(
            base_url + "/get_server_info", headers=get_auth_headers()
        )
        if server_info.status_code == 200:
            server_info_json = server_info.json()
            if "decode" in server_info_json:
                server_info_json = server_info_json["decode"][0]
            if (
                "internal_states" in server_info_json
                and server_info_json["internal_states"]
            ):
                accept_length = server_info_json["internal_states"][0].get(
                    "avg_spec_accept_length", None
                )
            else:
                accept_length = None
        else:
            accept_length = None
    else:
        accept_length = None

    # Compute metrics and print results
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
        accept_length=accept_length,
        plot_throughput=args.plot_throughput,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print(
        "{:<40} {:<10}".format(
            "Traffic request rate:", "trace" if use_trace_timestamps else request_rate
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Max request concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total input text tokens:", metrics.total_input_text))
    print(
        "{:<40} {:<10}".format("Total input vision tokens:", metrics.total_input_vision)
    )
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Peak concurrent requests:", metrics.max_concurrent_requests
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    if accept_length:
        print("{:<40} {:<10.2f}".format("Accept length:", accept_length))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("Max ITL (ms):", metrics.max_itl_ms))
    print("=" * 50)

    resp = requests.get(base_url + "/get_server_info", headers=get_auth_headers())
    server_info = resp.json() if resp.status_code == 200 else None

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # Arguments
            "tag": getattr(args, "tag", None),
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": "trace" if use_trace_timestamps else request_rate,
            "max_concurrency": max_concurrency,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            # Information
            "server_info": server_info,
            # Results
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_input_text_tokens": metrics.total_input_text,
            "total_input_vision_tokens": metrics.total_input_vision,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "total_throughput": metrics.total_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p95_itl_ms": metrics.p95_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "concurrency": metrics.concurrency,
            "accept_length": accept_length,
            "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
            "max_concurrent_requests": metrics.max_concurrent_requests,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "image":
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_"
                f"{args.random_output_len}_{args.image_count}imgs_"
                f"{args.image_resolution}.jsonl"
            )
        elif args.dataset_name.startswith("random"):
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"
            )

    result_details = {
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        if args.output_details:
            result_for_dump = result | result_details
        else:
            result_for_dump = result
        file.write(json.dumps(result_for_dump) + "\n")

    return result | result_details


def check_chat_template(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False


def set_global_args(args_: argparse.Namespace):
    """Set the global args."""
    global args
    args = args_


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set default value for max_concurrency if not present
    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    # Set default value for warmup_requests if not present
    if not hasattr(args, "warmup_requests"):
        args.warmup_requests = 1

    if not hasattr(args, "output_details"):
        args.output_details = False

    if not hasattr(args, "tokenize_prompt"):
        args.tokenize_prompt = False

    if not hasattr(args, "plot_throughput"):
        args.plot_throughput = False

    if not hasattr(args, "use_trace_timestamps"):
        args.use_trace_timestamps = False
    if not hasattr(args, "mooncake_slowdown_factor"):
        args.mooncake_slowdown_factor = 1.0

    if not hasattr(args, "mooncake_slowdown_factor"):
        args.mooncake_slowdown_factor = 1.0

    if not hasattr(args, "mooncake_num_rounds"):
        args.mooncake_num_rounds = 1

    if not hasattr(args, "served_model_name"):
        args.served_model_name = None

    if getattr(args, "print_requests", False):
        assert args.backend == "sglang-oai-chat"  # only support this now

    print(f"benchmark_args={args}")

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    if args.tokenize_prompt:
        assert (
            args.backend == "sglang"
        ), "`--tokenize-prompt` only compatible with `--backend sglang` currently"

    # Set url
    if args.port is None:
        args.port = {
            "sglang": 30000,
            "sglang-native": 30000,
            "sglang-oai": 30000,
            "lmdeploy": 23333,
            "vllm": 8000,
            "trt": 8000,
            "gserver": 9988,
            "truss": 8080,
        }.get(args.backend, 30000)

    model_url = (
        f"{args.base_url}/v1/models"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/models"
    )

    if args.backend in ["sglang", "sglang-native"]:
        api_url = (
            f"{args.base_url}/generate"
            if args.base_url
            else f"http://{args.host}:{args.port}/generate"
        )
    elif args.backend in ["sglang-oai", "vllm", "lmdeploy"]:
        api_url = (
            f"{args.base_url}/v1/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/completions"
        )
    elif args.backend in ["sglang-oai-chat", "vllm-chat", "lmdeploy-chat"]:
        api_url = (
            f"{args.base_url}/v1/chat/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/chat/completions"
        )
    elif args.backend == "trt":
        api_url = (
            f"{args.base_url}/v2/models/ensemble/generate_stream"
            if args.base_url
            else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
        )
        if args.model is None:
            print("Please provide a model using `--model` when using `trt` backend.")
            sys.exit(1)
    elif args.backend == "gserver":
        api_url = args.base_url if args.base_url else f"{args.host}:{args.port}"
        args.model = args.model or "default"
    elif args.backend == "truss":
        api_url = (
            f"{args.base_url}/v1/models/model:predict"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/models/model:predict"
        )
    base_url = (
        f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url
    )

    # Get model name
    if args.model is None:
        if args.backend == "truss":
            print(
                "Please provide a model with `--model` when using truss backend. e.g. --model meta-llama/Llama-3.1-8B-Instruct"
            )
            sys.exit(1)
        try:
            response = requests.get(model_url, headers=get_auth_headers())
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    if not check_chat_template(args.model):
        print(
            "\nWARNING It is recommended to use the `Chat` or `Instruct` model for benchmarking.\n"
            "Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly.\n"
        )

    if args.dataset_name in ["image", "mmmu"]:
        args.apply_chat_template = True
        assert (
            not args.tokenize_prompt
        ), "`--tokenize-prompt` not compatible with image dataset"

    if args.lora_request_distribution in ["distinct", "skewed"]:
        assert (
            args.lora_name is not None and len(args.lora_name) > 1
        ), "More than 1 LoRA adapter must be specified via --lora-name to use 'distinct' or 'skewed' request distribution."

    assert (
        args.lora_zipf_alpha > 1
    ), f"Got invalid value for --lora-zipf-alpha of {args.lora_zipf_alpha}. It must be greater than 1."

    print(f"{args}\n")

    # Read dataset
    backend = args.backend
    model_id = args.served_model_name or args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id)
    input_requests = get_dataset(args, tokenizer, model_id)

    # compatible with SimpleNamespace
    if not hasattr(args, "flush_cache"):
        args.flush_cache = False

    # Prepare LoRA arguments
    lora_request_distribution = (
        args.lora_request_distribution if args.lora_name is not None else None
    )

    lora_zipf_alpha = (
        args.lora_zipf_alpha
        if args.lora_name is not None and args.lora_request_distribution == "skewed"
        else None
    )

    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            lora_names=args.lora_name,
            lora_request_distribution=lora_request_distribution,
            lora_zipf_alpha=lora_zipf_alpha,
            extra_request_body=extra_request_body,
            profile=args.profile,
            pd_separated=args.pd_separated,
            flush_cache=args.flush_cache,
            warmup_requests=args.warmup_requests,
            use_trace_timestamps=args.use_trace_timestamps,
            mooncake_slowdown_factor=args.mooncake_slowdown_factor,
            mooncake_num_rounds=args.mooncake_num_rounds,
            profile_prefill_url=getattr(args, "profile_prefill_url", None),
            profile_decode_url=getattr(args, "profile_decode_url", None),
        )
    )


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


class LoRAPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [])
        for lora_name in values:
            getattr(namespace, self.dest).append(lora_name)


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "random",
            "random-ids",
            "generated-shared-prefix",
            "mmmu",
            "image",
            "mooncake",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        help="The name of the model as served by the serving service. If not set, this defaults to the value of --model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random and image dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random and image dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random and image dataset.",
    )
    # image dataset args
    parser.add_argument(
        "--image-count",
        type=int,
        default=1,
        help="Number of images per request (only available with the image dataset)",
    )
    parser.add_argument(
        "--image-resolution",
        type=str,
        default="1080p",
        help=(
            "Resolution of images for image dataset. "
            "Supports presets 4k/1080p/720p/360p or custom 'heightxwidth' (e.g., 1080x1920)."
        ),
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpeg",
        help=("Format of images for image dataset. " "Supports jpeg and png."),
    )
    parser.add_argument(
        "--image-content",
        type=str,
        default="random",
        help=("Content for images for image dataset. " "Supports random and blank."),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--use-trace-timestamps",
        action="store_true",
        help="Use timestamps from the trace file for request scheduling. Only valid for 'mooncake' dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--output-details", action="store_true", help="Output details of benchmarking."
    )
    parser.add_argument(
        "--print-requests",
        action="store_true",
        help="Print requests immediately during benchmarking. Useful to quickly realize issues.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--plot-throughput",
        action="store_true",
        help="Plot throughput and concurrent requests over time. Requires termplotlib and gnuplot.",
    )
    # TODO unify all these
    parser.add_argument(
        "--profile-activities",
        type=str,
        nargs="+",
        default=["CPU", "GPU"],
        choices=["CPU", "GPU", "CUDA_PROFILER"],
    )
    parser.add_argument("--profile-num-steps", type=int, default=None)
    parser.add_argument("--profile-by-stage", action="store_true", default=False)
    parser.add_argument("--profile-stages", nargs="+", default=None)
    parser.add_argument(
        "--lora-name",
        type=str,
        nargs="*",
        default=None,
        action=LoRAPathAction,
        help="The names of LoRA adapters. You can provide a list of names in the format {name} {name} {name}...",
    )
    parser.add_argument(
        "--lora-request-distribution",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "distinct",
            "skewed",
        ],
        help="What distribution to sample the LoRA adapters specified in --lora-name. Borrowed from the Punica paper. "
        "'distinct' distribution means selecting a new LoRA adapter for every request. "
        "'skewed' distribution follows the Zipf distribution, where the number of requests "
        "to model i specified in --lora-name is α times the number of requests for model i+1, "
        "where α > 1.",
    )
    parser.add_argument(
        "--lora-zipf-alpha",
        type=float,
        default=1.5,
        help="The parameter to use for the Zipf distribution when --lora-request-distribution='skewed'.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="",
        help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
    )
    parser.add_argument(
        "--pd-separated",
        action="store_true",
        help="Benchmark PD disaggregation server",
    )

    # Create a mutually exclusive group for profiling URLs
    # In PD separated mode, prefill and decode workers must be profiled separately
    profile_url_group = parser.add_mutually_exclusive_group()
    profile_url_group.add_argument(
        "--profile-prefill-url",
        type=str,
        nargs="*",
        default=None,
        help="URL(s) of the prefill worker(s) for profiling in PD separated mode. "
        "Can specify multiple URLs: --profile-prefill-url http://localhost:30000 http://localhost:30001. "
        "NOTE: Cannot be used together with --profile-decode-url. "
        "In PD separated mode, prefill and decode workers must be profiled separately.",
    )
    profile_url_group.add_argument(
        "--profile-decode-url",
        type=str,
        nargs="*",
        default=None,
        help="URL(s) of the decode worker(s) for profiling in PD separated mode. "
        "Can specify multiple URLs: --profile-decode-url http://localhost:30010 http://localhost:30011. "
        "NOTE: Cannot be used together with --profile-prefill-url. "
        "In PD separated mode, prefill and decode workers must be profiled separately.",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Flush the cache before running the benchmark",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of warmup requests to run before the benchmark",
    )
    parser.add_argument(
        "--tokenize-prompt",
        action="store_true",
        help="Use integer ids instead of string for inputs. Useful to control prompt lengths accurately",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--gsp-range-ratio",
        type=float,
        # WARN: The default 1.0 is for backward compatibility, and is different from the default 0.0 for random dataset
        default=1.0,
        help="Range of sampled ratio of input/output length, used only for gsp dataset.",
    )
    mooncake_group = parser.add_argument_group("mooncake dataset arguments")
    mooncake_group.add_argument(
        "--mooncake-slowdown-factor",
        type=float,
        default=1.0,
        help="Slowdown factor for replaying the mooncake trace. "
        "A value of 2.0 means the replay is twice as slow. "
        "NOTE: --request-rate is IGNORED in mooncake mode.",
    )
    mooncake_group.add_argument(
        "--mooncake-num-rounds",
        type=int,
        default=1,
        help="Number of conversation rounds for each session in the mooncake dataset. "
        "A value > 1 will enable true multi-turn session benchmarking.",
    )
    mooncake_group.add_argument(
        "--mooncake-workload",
        type=str,
        default="conversation",
        choices=[
            "mooncake",
            "conversation",
            "synthetic",
            "toolagent",
        ],
        help="Underlying workload for the mooncake dataset.",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="The tag to be dumped to output."
    )
    args = parser.parse_args()
    run_benchmark(args)
