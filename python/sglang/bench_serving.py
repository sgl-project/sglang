# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

"""
Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate-range 1,2,4,8,16,32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --multi
"""

import argparse
import asyncio
import json
import os
import pickle
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import csv

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from .data_processing import(
    get_dataset_from_openai,
    get_mooncake_trace,
    SampleOutput
)
from datasets import load_dataset
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

global args


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


# trt llm not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
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
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = response.reason or ""
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

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            "lora_path": request_func_input.lora_name,
            **request_func_input.extra_request_body,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_truss(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
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
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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
                            if data["choices"][0]["delta"]["content"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["delta"]["content"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ""
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

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": request_func_input.output_len,
                "ignore_eos": not args.disable_ignore_eos,
            },
            "stream": not args.disable_stream,
            "lora_path": request_func_input.lora_name,
            **request_func_input.extra_request_body,
        }
        headers = {}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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
                        # print(chunk_bytes)

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text = data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_gserver(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError()

async def async_request_dynamo(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": "Meta-Llama-3-8B-Instruct",
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            "lora_path": request_func_input.lora_name,
            **request_func_input.extra_request_body,
        }
        headers = {"Content-Type": "application/json"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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
                    # if output.ttft == 0.0:
                        # print("=================================")
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_loongserve(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": request_func_input.output_len,
            }
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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

                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


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
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def get_dataset(args, tokenizer):
    if args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )
        input_length = [r[1] for r in input_requests]
        output_length = [r[2] for r in input_requests]
        print(f"max input length: {max(input_length)} min input length: {min(input_length)}, mean input length: {np.mean(input_length)}")
        print(f"max output length: {max(output_length)} min output length: {min(output_length)}, mean output length: {np.mean(output_length)}")
    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
        )
    elif args.dataset_name == "generated-shared-prefix":
        input_requests = sample_generated_shared_prefix_requests(
            num_groups=args.gen_num_groups,
            prompts_per_group=args.gen_prompts_per_group,
            system_prompt_len=args.gen_system_prompt_len,
            question_len=args.gen_question_len,
            output_len=args.gen_output_len,
            tokenizer=tokenizer,
        )
    elif args.dataset_name in ["loogle", "leval", "sharegpt-prefix"]:
        input_requests = sample_openai_requests(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            tokenizer=tokenizer
        )
        input_length = [r[1] for r in input_requests]
        output_length = [r[2] for r in input_requests]
        print(f"max input length: {max(input_length)} min input length: {min(input_length)}, mean input length: {np.mean(input_length)}")
        print(f"max output length: {max(output_length)} min output length: {min(output_length)}, mean output length: {np.mean(output_length)}")
    elif args.dataset_name == "mooncake":
        input_requests = sample_mooncake_requests(
            model_name=args.model,
            num_requests=args.num_prompts,
            dataset_path=args.dataset_path,
        )
    elif args.dataset_name == "synthesis":
        input_requests = sample_synthesis_requests(
            num_requests=args.num_prompts, input_length=args.synthesis_input_length, output_length=args.synthesis_output_length,tokenizer=tokenizer
        )
    elif args.dataset_name == "synthesis_shuffle":
        input_requests = sample_synthesis_requests(
            num_requests=args.num_prompts, input_length=args.synthesis_input_length, output_length=args.synthesis_output_length,
        )
    elif args.dataset_name == "azure":
        input_requests = sample_azure_requests(
            num_requests=args.num_prompts,
            data_path="/home/whcui/AzurePublicDataset/data/AzureLLMInferenceTrace_code.csv",
        )
    elif args.dataset_name == "mixed":
        input_requests = sample_mixed_requests(
            num_requests=args.num_prompts,
            loogle_path=args.dataset_path,
            long_req_ratio=args.long_req_ratio,
            tokenizer=tokenizer
        )
    elif args.dataset_name == "open-thoughts":
        input_requests = sample_open_thoughts_requests(
            num_prompts=args.num_prompts,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return input_requests


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_sglang_generate,
    "sglang-native": async_request_sglang_generate,
    "sglang-oai": async_request_openai_completions,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "trt": async_request_trt_llm,
    "gserver": async_request_gserver,
    "truss": async_request_truss,
    "dynamo": async_request_dynamo,
    "loongserve": async_request_loongserve
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
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
    mean_ttft_per_token_ms: float
    median_ttft_per_token_ms: float
    std_ttft_per_token_ms: float
    p99_ttft_per_token_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float


SHAREGPT_URL = "https://www.modelscope.cn/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
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


def sample_openai_requests(
    dataset_name: str,
    dataset_path: str,
    num_prompts: int,
    fixed_output_len: int = None,
    max_sum_len: int = None,
    tokenizer: PreTrainedTokenizerBase = None,
) -> List[Tuple[str, int, int]]:
    if dataset_name == "sharegpt-prefix":
        dataset_name = "sharegpt"
    conversations = get_dataset_from_openai(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        num_prompts=num_prompts,
        fixed_output_len=fixed_output_len,
        max_sum_len=max_sum_len,
        tokenizer=tokenizer,
    )
    requests = []
    for conversation in conversations:
        requests.extend(conversation)
    return requests

def sample_mooncake_requests(
    model_name: str,
    num_requests: int,
    dataset_path: str
):
    conversations, _ = get_mooncake_trace(
        model_name=model_name,
        dataset_path=dataset_path,
        max_sum_len=100000,
    )
    input_requests = []
    for conversation in conversations:
        input_requests.extend(conversation)
    input_requests = input_requests[:num_requests]
    return input_requests

def sample_synthesis_requests(
        num_requests: int,
        input_length: int,
        output_length:int,
        tokenizer:PreTrainedTokenizer) -> List[Tuple[str, int, int]]:
    sample_results: List[Tuple[str, int, int]] = []
    gap = 0
    for i in range(num_requests):
        input_ids = [i + 200 + gap for _ in range(input_length)]
        while(len(tokenizer.tokenize(tokenizer.decode(input_ids))) != input_length):
            gap += 1
            input_ids = [i + 200 + gap for _ in range(input_length)]
        assert len(tokenizer.tokenize(tokenizer.decode(input_ids))) == input_length
        sample_results.append((tokenizer.decode(input_ids), input_length, output_length))
    return sample_results

def sample_azure_requests(num_requests: int, data_path: str) -> List[Tuple[str, int, int]]:
    sample_results: List[Tuple[str, int, int]] = []
    with open(data_path) as f:
        reader = csv.reader(f)
        _ = next(reader)
        for _ in range(num_requests):
            row = next(reader)
            input_len = int(row[1])
            input_len *= 8
            output_len = int(row[2])
            sample_results.append((str(input_len), input_len, output_len))
    return sample_results

def sample_mixed_requests(num_requests: int, loogle_path:str, long_req_ratio: int, tokenizer: PreTrainedTokenizer) -> List[Tuple[str, int, int]]:
    long_num_req = num_requests * long_req_ratio
    short_num_req = num_requests - long_num_req
    conversations = get_dataset_from_openai(
        dataset_name="loogle",
        dataset_path=loogle_path,
        num_prompts=long_num_req,
        fixed_output_len=None,
        max_sum_len=None,
        tokenizer=tokenizer,
    )
    long_requests = [con[0] for con in conversations]
    short_requests = sample_sharegpt_requests(
        "/",
        short_num_req,
        tokenizer=tokenizer
    )
    sample_requests = long_requests + short_requests
    random.shuffle(sample_requests)
    return sample_requests

def sample_open_thoughts_requests(
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase = None,
) -> List[Tuple[str, int, int]]:
    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    input_requests = []
    for i in range(num_prompts):
        if i >= len(ds):
            break
        prompt = ds[i]["system"] + ds[i]["conversations"][0]["value"]
        completion = ds[i]["conversations"][1]["value"]
        prompt_token_ids = tokenizer.encode(prompt)
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        input_requests.append((prompt, prompt_len, output_len))
    return input_requests  

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or (
            prompt_len + output_len > 2048 and fixed_output_len is None
        ):
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    print(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    return filtered_dataset


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
) -> List[Tuple[str, int, int]]:

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    if True:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[Tuple[str, int, int]] = []
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
            prompt = tokenizer.decode(input_ids)
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % tokenizer.vocab_size
                    for j in range(input_lens[i])
                ]
            )
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def get_gen_prefix_cache_path(args, tokenizer):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    # Create a unique cache filename based on the generation parameters
    cache_key = (
        f"gen_prefix_{args.gen_num_groups}_{args.gen_prompts_per_group}_"
        f"{args.gen_system_prompt_len}_{args.gen_question_len}_{args.gen_output_len}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(args, tokenizer)

    # Try to load from cache first
    if cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("\nGenerating new input data...")

    # Generate system prompts for each group
    system_prompts = []
    for _ in range(num_groups):
        system_prompt = gen_prompt(tokenizer, system_prompt_len)
        system_prompts.append(system_prompt)

    # Generate questions
    questions = []
    for _ in range(num_groups * prompts_per_group):
        question = gen_prompt(tokenizer, question_len)
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
            question = questions[group_idx * prompts_per_group + prompt_idx]
            full_prompt = f"{system_prompt}\n\n{question}"
            prompt_len = len(tokenizer.encode(full_prompt))

            input_requests.append((full_prompt, prompt_len, output_len))
            total_input_tokens += prompt_len
            total_output_tokens += output_len

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
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)
        
async def get_request_from_trace(
    reqs: List[Tuple[str, int, int]],
    traces: list[float],
    trace_start_time: float
) -> AsyncGenerator[Tuple[str, int, int], None]:
    assert len(reqs) == len(traces)
    num_requests = len(reqs)
    start_timestamp = time.time()
    for idx in range(num_requests):
        target = start_timestamp + traces[idx] - trace_start_time * args.trace_rate_scale
        time_to_sleep = target - time.time()
        if time_to_sleep > 0:
            await asyncio.sleep(time_to_sleep)
        yield reqs[idx]
        

def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
    request_rate: int,
    input: int = 0,
    output: int = 0,
    traces: list[float] = None
) -> Tuple[BenchmarkMetrics, List[int]]:
    global args
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    ttfts_per_token: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            ttfts_per_token.append(outputs[i].ttft/outputs[i].prompt_len)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    itl_p99 = np.percentile(itls or 0, 99) * 1000
    # if itl_p99 > 70 and itl_p99 < 90 :
    # if ttfts[-1]<1000:
    if backend == "sglang":
        if args.chunk_size:
            backend = "chunked"
        else:
            backend = "yoda"
    if args.dataset_name == "mooncake-trace":
        if "conv" in args.dataset_path:
            dataset_name = "mooncake-trace-conv"
        else:
            dataset_name = "mooncake-trace-tool"
    else:
        dataset_name = args.dataset_name
    model_size = '8B' if '8B' in args.model else '70B'
    with open(f"/workspace/source/PD-Multiplexing/res/{model_size}/{dataset_name}/{backend}.jsonl", "a+") as f:
        data = {
            "chunk size": args.chunk_size,
            "batch limit": args.batch_limit,
            "capacity limit": args.capacity_limit,
            "request_rate": request_rate,
            "trace_rate_scale": args.trace_rate_scale,
            "trace_start_time": args.trace_start_time,
            "trace_end_time": args.trace_end_time,
            "completed": completed,
            "ttft mean":"%.2f" % (np.mean(ttfts or 0) * 1000),
            "ttft p99":"%.2f" % (np.percentile(ttfts or 0, 99) * 1000),
            "tpot mean":"%.2f" % (np.mean(tpots or 0) * 1000),
            "tpot p99":"%.2f" % (np.percentile(tpots or 0, 99) * 1000),
            "itl mean":"%.2f" % (np.mean(itls or 0) * 1000),
            "itl p99":"%.2f" % (np.percentile(itls or 0, 99) * 1000),
            "ttfts":[ttft * 1000 for ttft in ttfts],
            "topts":[tpot * 1000 for tpot in tpots],
            "itls": [[i * 1000 for i in output.itl] for output in outputs if output.success],
            "timestamps":[t - args.trace_start_time * args.trace_rate_scale for t in traces] if traces else None
        }
        json_line = json.dumps(data)
        f.write(json_line + "\n")
    if not os.path.exists(f"/workspace/source/PD-Multiplexing/res/{model_size}/{dataset_name}/{backend}_general.csv"):
            with open(f"/workspace/source/PD-Multiplexing/res/{model_size}/{dataset_name}/{backend}_general.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ["chunk size", "batch limit", "capacity limit", "request rate", "trace_rate_scale", "trace_start_time", "trace_end_time","completed", "ttft mean", "ttft p99", "tpot mean","tpot p99", "itl mean", "itl p99"]
                writer.writerow(headers)
    with open(f"/workspace/source/PD-Multiplexing/res/{model_size}/{dataset_name}/{backend}_general.csv", "a+") as f:
        writer = csv.writer(f)
        row = [
            args.chunk_size,
            args.batch_limit,
            args.capacity_limit,
            request_rate,
            args.trace_rate_scale,
            args.trace_start_time,
            args.trace_end_time,
            completed,
            "%.2f" % (np.mean(ttfts or 0) * 1000),
            "%.2f" % (np.percentile(ttfts or 0, 99) * 1000),
            "%.2f" % (np.mean(tpots or 0) * 1000),
            "%.2f" % (np.percentile(tpots or 0, 99) * 1000),
            "%.2f" % (np.mean(itls or 0) * 1000),
            "%.2f" % (np.percentile(itls or 0, 99) * 1000),
        ]
        writer.writerow(row)
    
    

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
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
        mean_ttft_per_token_ms=np.mean(ttfts_per_token or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_per_token_ms=np.median(ttfts_per_token or 0) * 1000,
        std_ttft_per_token_ms=np.std(ttfts_per_token or 0) * 1000,
        p99_ttft_per_token_ms=np.percentile(ttfts_per_token or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    lora_name: str,
    extra_request_body: Dict[str, Any],
    profile: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    # print("Starting initial single prompt test run...")
    # test_prompt, test_prompt_len, test_output_len = input_requests[0]
    # test_input = RequestFuncInput(
    #     model=model_id,
    #     prompt=test_prompt,
    #     api_url=api_url,
    #     prompt_len=test_prompt_len,
    #     output_len=test_output_len,
    #     lora_name=lora_name,
    #     extra_request_body=extra_request_body,
    # )
    # test_output = await request_func(request_func_input=test_input)
    # if not test_output.success:
    #     raise ValueError(
    #         "Initial test run failed - Please make sure benchmark arguments "
    #         f"are correctly specified. Error: {test_output.error}"
    #     )
    # else:
    #     print("Initial test run completed. Starting main benchmark run...")

    time.sleep(1.5)

    print("warm up ...")
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input)
            )
        )
        if len(tasks) >= 10:
            break
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    print("warm up done ...")
    os.system(f"bash /workspace/source/PD-Multiplexing/python/sglang/flush_sglang.sh 30000")
    time.sleep(5)    
    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
        request_rate=request_rate,
        input=args.synthesis_input_length,
        output=args.synthesis_output_length,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max reqeuest concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
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
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
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
    print("{s:{c}^{n}}".format(s="Time to First Token Per token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT Per token (ms):", metrics.mean_ttft_per_token_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT Per token (ms):", metrics.median_ttft_per_token_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT Per token (ms):", metrics.p99_ttft_per_token_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "output_throughput": metrics.output_throughput,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "duration": benchmark_duration,
            "completed": metrics.completed,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    # if args.output_file:
    #     output_file_name = args.output_file
    # else:
    #     now = datetime.now().strftime("%m%d")
    #     if args.dataset_name == "random":
    #         output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
    #     else:
    #         output_file_name = f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"

    # Append results to a JSONL file
    # with open(output_file_name, "a") as file:
    #     file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
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
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }
    return result

async def benchmark_trace(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    reqs: List[Tuple[str, int, int]],
    traces: List[float],
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    lora_name: str,
    extra_request_body: Dict[str, Any],
    profile: bool,
    trace_rate_scale: float = 1,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len = reqs[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        lora_name=lora_name,
        extra_request_body=extra_request_body,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    time.sleep(1.5)

    print("warm up ...")
    tasks: List[asyncio.Task] = []
    async for request in get_request_from_trace(reqs, traces, args.trace_start_time):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input)
            )
        )
        if len(tasks) >= 10:
            break
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    os.system(f"bash /workspace/source/PD-Multiplexing/python/sglang/flush_sglang.sh 30000")
    print("warm up done ...")
    time.sleep(5)

    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    pbar = None if disable_tqdm else tqdm(total=len(reqs))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request_from_trace(reqs, traces, args.trace_start_time):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, output_lens = calculate_metrics(
        input_requests=reqs,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
        request_rate=None,
        input=args.synthesis_input_length,
        output=args.synthesis_output_length,
        traces=traces
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Trace rate scale:", trace_rate_scale))
    print(
        "{:<40} {:<10}".format(
            "Max reqeuest concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
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
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
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
    print("{s:{c}^{n}}".format(s="Time to First Token Per token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT Per token (ms):", metrics.mean_ttft_per_token_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT Per token (ms):", metrics.median_ttft_per_token_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT Per token (ms):", metrics.p99_ttft_per_token_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "trace rate scale": trace_rate_scale,
            "max_concurrency": max_concurrency,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "output_throughput": metrics.output_throughput,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "duration": benchmark_duration,
            "completed": metrics.completed,
        }
    else:
        print(f"Error running benchmark for trace rate scale: {trace_rate_scale}")
        print("-" * 30)

    # Determine output file name
    # if args.output_file:
    #     output_file_name = args.output_file
    # else:
    #     now = datetime.now().strftime("%m%d")
    #     if args.dataset_name == "random":
    #         output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
    #     else:
    #         output_file_name = f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"

    # Append results to a JSONL file
    # with open(output_file_name, "a") as file:
    #     file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
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
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }
    return result

def parse_request_rate_range(request_rate_range):
    if len(request_rate_range.split(",")) == 3:
        start, stop, step = map(int, request_rate_range.split(","))
        return list(range(start, stop, step))
    else:
        return list(map(int, request_rate_range.split(",")))


def check_chat_template(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set default value for max_concurrency if not present
    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

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
            "dynamo":8000,
            "loongserve":8400,
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
        if args.synthesis:
            api_url = (
                f"{args.base_url}/generate_synthesis"
                if args.base_url
                else f"http://{args.host}:{args.port}/generate_synthesis"
            )
    elif args.backend == "loongserve":
        api_url = f"http://{args.host}:{args.port}/generate_stream"
    elif args.backend in ["sglang-oai", "vllm", "lmdeploy", "dynamo"]:
        api_url = (
            f"{args.base_url}/v1/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/completions"
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
            response = requests.get(model_url)
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

    print(f"{args}\n")

    # Read dataset
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id)

    if args.dataset_name == "mooncake-trace":
        conversations, traces = get_mooncake_trace(model_name=model_id, dataset_path=args.dataset_path, max_sum_len=100000)
        input_requests = []
        for conversation in conversations:
            input_requests.extend(conversation)
        scale = args.trace_rate_scale

        selected_requests = [r for r,t in zip(input_requests, traces) if  t >= args.trace_start_time and (args.trace_end_time < 0 or t <= args.trace_end_time)]
        selected_traces = [t for t in traces if t >= args.trace_start_time and (args.trace_end_time < 0 or t <= args.trace_end_time)]
        selected_traces = [trace * scale for trace in selected_traces]    
    else:    
        input_requests = get_dataset(args, tokenizer)

    if args.dataset_name == "mooncake-trace":
        return asyncio.run(
            benchmark_trace(
                backend=backend,
                api_url=api_url,
                base_url=base_url,
                model_id=model_id,
                tokenizer=tokenizer,
                reqs=selected_requests,
                traces=selected_traces,
                max_concurrency=args.max_concurrency,
                disable_tqdm=args.disable_tqdm,
                lora_name=args.lora_name,
                extra_request_body=extra_request_body,
                profile=args.profile,
                trace_rate_scale=scale,
            )
        )
    elif not args.multi:
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
                lora_name=args.lora_name,
                extra_request_body=extra_request_body,
                profile=args.profile,
            )
        )
    else:
        # Benchmark multiple rps. TODO: use a fixed duration to compute num_prompts
        request_rates = parse_request_rate_range(args.request_rate_range)

        for rate in request_rates:
            asyncio.run(
                benchmark(
                    backend=backend,
                    api_url=api_url,
                    base_url=base_url,
                    model_id=model_id,
                    tokenizer=tokenizer,
                    input_requests=input_requests,
                    request_rate=rate,
                    max_concurrency=args.max_concurrency,
                    disable_tqdm=args.disable_tqdm,
                    lora_name=args.lora_name,
                    extra_request_body=extra_request_body,
                    profile=args.profile,
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
        choices=["sharegpt", "random", "generated-shared-prefix", "synthesis", "synthesis_shuffle", "azure", "loogle", "leval","sharegpt-prefix", "mooncake-trace", "mooncake","mixed", "open-thoughts"],
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
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
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
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use request rate range rather than single value.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step. Default is 2,34,2. It also supports a list of request rates, requiring the parameters to not equal three.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
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

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gen-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gen-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gen-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gen-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gen-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help="The name of LoRA adapter",
    )
    parser.add_argument(
        "--synthesis",
        action="store_true",
        help="request with sythesis propmt",
    )
    group.add_argument(
        "--synthesis-input-length",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--synthesis-output-length",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--batch-limit",
        type=int,
        default=0,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--capacity-limit",
        type=str,
        default=0,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--prefill-sm-percentage",
        type=float,
        default=0.7,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--decode-sm-percentage",
        type=float,
        default=0.3,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--long-req-ratio",
        type=float,
        default=0.1
    )
    group.add_argument(
        "--trace-rate-scale",
        type=float,
        default=1.0,
    )
    group.add_argument(
        "--trace-start-time",
        type=float,
        default=0.0
    )
    group.add_argument(
        "--trace-end-time",
        type=float,
        default=-1
    )
    
    args = parser.parse_args()
    run_benchmark(args)
