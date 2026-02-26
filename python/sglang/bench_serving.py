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
import copy
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import traceback
import uuid
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang.benchmark.datasets import DatasetRow, get_dataset
from sglang.benchmark.datasets.mooncake import get_mooncake_request_over_time
from sglang.benchmark.utils import (
    get_tokenizer,
    parse_custom_headers,
    remove_prefix,
    set_ulimit,
)

_ROUTING_KEY_HEADER = "X-SMG-Routing-Key"

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
    prompt: Union[str, List[str], List[Dict[str, str]]]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]
    timestamp: Optional[float] = None
    routing_key: Optional[str] = None


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


def get_auth_headers() -> Dict[str, str]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return {"Authorization": f"Bearer {openai_api_key}"}
    else:
        api_key = os.environ.get("API_KEY")
        if api_key:
            return {"Authorization": f"{api_key}"}
        return {}


def get_request_headers() -> Dict[str, str]:
    headers = get_auth_headers()
    if h := getattr(args, "header", None):
        headers.update(parse_custom_headers(h))
    return headers


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
        # Build payload with defaults that can be overridden by extra_request_body
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
        }

        # Add temperature default only if not specified in extra_request_body
        if "temperature" not in request_func_input.extra_request_body:
            payload["temperature"] = 0.0

        # Add ignore_eos default only if not specified in extra_request_body
        if "ignore_eos" not in request_func_input.extra_request_body:
            payload["ignore_eos"] = not args.disable_ignore_eos

        # Merge in extra parameters - these will override defaults if present
        payload.update(request_func_input.extra_request_body)

        # hack to accommodate different LoRA conventions between SGLang and vLLM.
        if request_func_input.lora_name:
            payload["model"] = request_func_input.lora_name
            payload["lora_path"] = request_func_input.lora_name

        if request_func_input.image_data:
            payload.update({"image_data": request_func_input.image_data})

        headers = get_request_headers()
        if request_func_input.routing_key:
            headers[_ROUTING_KEY_HEADER] = request_func_input.routing_key

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

    if isinstance(request_func_input.prompt, list):
        messages = request_func_input.prompt
    elif request_func_input.image_data:
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
        # Build payload with defaults that can be overridden by extra_request_body
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "max_completion_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
        }

        # Add temperature default only if not specified in extra_request_body
        if "temperature" not in request_func_input.extra_request_body:
            payload["temperature"] = 0.0

        # Add ignore_eos default only if not specified in extra_request_body
        # Default to False for more realistic behavior (respect EOS tokens)
        if "ignore_eos" not in request_func_input.extra_request_body:
            payload["ignore_eos"] = not args.disable_ignore_eos

        # Merge in extra parameters (tools, temperature, top_p, etc.)
        # These will override defaults if present
        payload.update(request_func_input.extra_request_body)

        # hack to accommodate different LoRA conventions between SGLang and vLLM.
        if request_func_input.lora_name:
            payload["model"] = request_func_input.lora_name
            payload["lora_path"] = request_func_input.lora_name

        headers = get_request_headers()
        if request_func_input.routing_key:
            headers[_ROUTING_KEY_HEADER] = request_func_input.routing_key

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
        headers = get_request_headers()

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
            "return_routed_experts": args.return_routed_experts,
            "logprob_start_len": -1,
            **request_func_input.extra_request_body,
        }

        # Add image data if available (list of image urls/base64)
        if request_func_input.image_data:
            payload["image_data"] = request_func_input.image_data

        headers = get_request_headers()
        if request_func_input.routing_key:
            headers[_ROUTING_KEY_HEADER] = request_func_input.routing_key

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
            if api_url.endswith("/start_profile"):
                num_steps = getattr(args, "profile_num_steps", None)
                profile_by_stage = getattr(args, "profile_by_stage", None)
                if profile_by_stage and num_steps is None:
                    num_steps = 5

                output_dir = getattr(args, "profile_output_dir", None)
                if output_dir is None:
                    output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
                output_dir = Path(os.path.abspath(os.path.normpath(output_dir))) / str(
                    time.time()
                )
                output_dir.mkdir(exist_ok=True, parents=True)
                output_dir = str(output_dir)

                body = {
                    "activities": getattr(args, "profile_activities", []),
                    "num_steps": num_steps,
                    "profile_by_stage": profile_by_stage,
                    "profile_stages": getattr(args, "profile_stages", None),
                    "output_dir": output_dir,
                    "profile_prefix": getattr(args, "profile_prefix", None),
                }
            else:
                # stop_profile doesn't need any parameters
                body = {}
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
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    max_output_tokens_per_s: float = 0.0
    max_concurrent_requests: int = 0


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
    input_requests: Optional[List[DatasetRow]],
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
            if input_requests is not None:
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
        p90_e2e_latency_ms=np.percentile(e2e_latencies, 90) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )

    return metrics, output_lens


MULTI_TURN_BACKENDS = {"sglang-oai-chat", "vllm-chat", "lmdeploy-chat"}


def wrap_multi_turn_request_func(request_func: Callable, backend: str) -> Callable:
    assert (
        backend in MULTI_TURN_BACKENDS
    ), f"Multi-turn only supports chat backends: {MULTI_TURN_BACKENDS}, got {backend}"

    async def f(
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> List[RequestFuncOutput]:
        prompts: List[str] = request_func_input.prompt
        prev_messages: List[Dict[str, str]] = []
        outputs = []

        for round_index in range(len(prompts)):
            prev_messages.append({"role": "user", "content": prompts[round_index]})

            inner_input = replace(
                copy.deepcopy(request_func_input), prompt=copy.deepcopy(prev_messages)
            )
            output = await request_func(
                inner_input, pbar=pbar if round_index == len(prompts) - 1 else None
            )
            outputs.append(output)

            prev_messages.append(
                {"role": "assistant", "content": output.generated_text}
            )

        return outputs

    return f


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

    # Check for multi-turn: prompt is a list of strings (not OpenAI messages dicts)
    # Multi-turn format: ["turn1", "turn2", ...] - list of strings
    # OpenAI format: [{"role": "user", "content": "..."}, ...] - list of dicts
    first_prompt = input_requests[0].prompt
    is_multi_turn = (
        isinstance(first_prompt, list)
        and len(first_prompt) > 0
        and isinstance(first_prompt[0], str)
    )
    if is_multi_turn:
        request_func = wrap_multi_turn_request_func(request_func, backend=backend)

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
    if is_multi_turn:
        warmup_outputs = [x for output in warmup_outputs for x in output]

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

        # Merge global extra_request_body with per-request extras
        # Per-request parameters take precedence over global ones
        merged_extra_body = {**extra_request_body, **request.extra_request_body}

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
            lora_name=lora_name,
            image_data=request.image_data,
            extra_request_body=merged_extra_body,
            timestamp=request.timestamp,
            routing_key=request.routing_key,
        )

        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    if is_multi_turn:
        outputs = [x for output in outputs for x in output]

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
        input_requests=None if is_multi_turn else input_requests,
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
    if args.dataset_name in ["image", "mmmu"]:
        print(
            "{:<40} {:<10}".format(
                "Total input vision tokens:", metrics.total_input_vision
            )
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
    print(
        "{:<40} {:<10.2f}".format("P90 E2E Latency (ms):", metrics.p90_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms)
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
            "p90_e2e_latency_ms": metrics.p90_e2e_latency_ms,
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
            "custom",
            "openai",
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
        "--random-image-count",
        action="store_true",
        help="Enable Random Image Count",
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
    parser.add_argument(
        "--return-routed-experts",
        action="store_true",
        help="Return routed experts.",
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
        "--profile-output-dir",
        type=str,
        default=None,
        help="Output directory for profile traces.",
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default=None,
        help="Prefix for profile trace filenames.",
    )
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
    group.add_argument(
        "--gsp-fast-prepare",
        action="store_true",
        help="Speedup preparing by removing statistics computation, which will make some output statistics inaccurate but suitable for pressure tests.",
    )
    group.add_argument(
        "--gsp-send-routing-key",
        action="store_true",
        help="Send routing key in requests via X-SMG-Routing-Key header. Requests with the same prefix share the same routing key.",
    )
    group.add_argument(
        "--gsp-num-turns",
        type=int,
        default=1,
        help="Number of turns for multi-turn conversations. If > 1, each prompt becomes a list of questions sharing the same system prefix.",
    )
    group.add_argument(
        "--gsp-ordered",
        action="store_true",
        help="Keep requests in order without shuffling. By default, requests are shuffled randomly.",
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
    parser.add_argument(
        "--header",
        type=str,
        nargs="+",
        default=None,
        help="Custom HTTP headers in Key=Value format. Example: --header MyHeader=MY_VALUE MyAnotherHeader=myanothervalue",
    )
    args = parser.parse_args()
    run_benchmark(args)
