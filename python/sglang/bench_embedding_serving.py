"""
Benchmark online embedding serving throughput with dynamic requests.

Usage:
python3 -m sglang.bench_embedding_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_embedding_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field, fields
from datetime import datetime
from json import JSONDecodeError
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.bench_serving import (
    get_dataset,
    get_tokenizer,
    set_ulimit,
    _get_bool_env_var,
    remove_prefix,
    get_auth_headers,
    _create_bench_client_session,
    DatasetRow,
    check_chat_template,
    async_request_profile,
    get_request,
)

ASSISTANT_SUFFIX = "Assistant:"


@dataclass
class BenchArgs:
    backend: str = "sglang"
    base_url: Optional[str] = None
    host: str = "0.0.0.0"
    port: Optional[int] = None
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    model: Optional[str] = None
    tokenizer: Optional[str] = None

    # compatibility for openai apis
    encoding_format: str = "float"
    dimensions: Optional[int] = None

    # ShareGPT dataset args
    sharegpt_output_len: Optional[int] = None
    sharegpt_context_len: Optional[int] = None
    # random dataset args
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    # random-image dataset args
    random_image_num_images: int = 1
    random_image_resolution: str = "1080p"

    seed: int = 1
    num_prompts: int = 1000
    max_concurrency: Optional[int] = None
    request_rate: float = float("inf")
    output_file: Optional[str] = None
    output_details: bool = False
    disable_tqdm: bool = False
    disable_ignore_eos: bool = False
    extra_request_body: Optional[str] = None
    apply_chat_template: bool = False
    profile: bool = False
    prompt_suffix: str = ""
    pd_separated: bool = False
    flush_cache: bool = False
    warmup_requests: int = 1
    tokenize_prompt: bool = False

    # generated-shared-prefix dataset arguments
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
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
            default=BenchArgs.base_url,
            help="Server or API base url if not using http host and port.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=BenchArgs.host,
            help="Default host is 0.0.0.0.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=BenchArgs.port,
            help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=[
                "sharegpt",
                "random",
                "random-ids",
                "generated-shared-prefix",
                "mmmu",
                "random-image",
            ],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=BenchArgs.model,
            help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=BenchArgs.tokenizer,
            help="Name or path of the tokenizer. If not set, using the model conf.",
        )
        parser.add_argument(
            "--encoding-format",
            type=str,
            default=BenchArgs.encoding_format,
            choices=["float", "base64"],
            help="The format to return the embeddings in. Can be either float or base64. NOTE: sglang does not support now",
        )
        parser.add_argument(
            "--dimensions",
            type=int,
            default=BenchArgs.dimensions,
            help="The number of dimensions the resulting output embeddings should have. NOTE: sglang does not support now",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length, used only for random dataset.",
        )
        # random-image dataset args
        parser.add_argument(
            "--random-image-num-images",
            type=int,
            default=BenchArgs.random_image_num_images,
            help="Number of images per request (only available with the random-image dataset)",
        )
        parser.add_argument(
            "--random-image-resolution",
            type=str,
            default=BenchArgs.random_image_resolution,
            help=(
                "Resolution of random images for random-image dataset. "
                "Supports presets 4k/1080p/720p/360p or custom 'heightxwidth' (e.g., 1080x1920)."
            ),
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--max-concurrency",
            type=int,
            default=BenchArgs.max_concurrency,
            help="Maximum number of concurrent requests. This can be used "
            "to help simulate an environment where a higher level component "
            "is enforcing a maximum number of concurrent requests. While the "
            "--request-rate argument controls the rate at which requests are "
            "initiated, this argument will control how many are actually allowed "
            "to execute at a time. This means that when used in combination, the "
            "actual request rate may be lower than specified with --request-rate, "
            "if the server is not processing requests fast enough to keep up.",
        )
        parser.add_argument(
            "--request-rate",
            type=float,
            default=BenchArgs.request_rate,
            help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
            "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default=BenchArgs.output_file,
            help="Output JSONL file name.",
        )
        parser.add_argument(
            "--output-details",
            action="store_true",
            default=BenchArgs.output_details,
            help="Output details of benchmarking.",
        )
        parser.add_argument(
            "--disable-tqdm",
            action="store_true",
            default=BenchArgs.disable_tqdm,
            help="Specify to disable tqdm progress bar.",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            default=BenchArgs.disable_ignore_eos,
            help="Disable ignoring EOS.",
        )
        parser.add_argument(
            "--extra-request-body",
            metavar='{"key1": "value1", "key2": "value2"}',
            type=str,
            default=BenchArgs.extra_request_body,
            help="Append given JSON object to the request payload. You can use this to specify"
            "additional generate params like sampling params.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            default=BenchArgs.apply_chat_template,
            help="Apply chat template",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            default=BenchArgs.profile,
            help="Use Torch Profiler. The endpoint must be launched with SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default=BenchArgs.prompt_suffix,
            help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
        )
        parser.add_argument(
            "--pd-separated",
            action="store_true",
            default=BenchArgs.pd_separated,
            help="Benchmark PD disaggregation server",
        )
        parser.add_argument(
            "--flush-cache",
            action="store_true",
            default=BenchArgs.flush_cache,
            help="Flush the cache before running the benchmark",
        )
        parser.add_argument(
            "--warmup-requests",
            type=int,
            default=BenchArgs.warmup_requests,
            help="Number of warmup requests to run before the benchmark",
        )
        parser.add_argument(
            "--tokenize-prompt",
            action="store_true",
            default=BenchArgs.tokenize_prompt,
            help="Use integer ids instead of string for inputs. Useful to control prompt lengths accurately",
        )

        group = parser.add_argument_group("generated-shared-prefix dataset arguments")
        group.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of system prompt groups for generated-shared-prefix dataset",
        )
        group.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per system prompt group for generated-shared-prefix dataset",
        )
        group.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="Target length in tokens for system prompts in generated-shared-prefix dataset",
        )
        group.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Target length in tokens for questions in generated-shared-prefix dataset",
        )
        group.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def _default_port(self):
        if self.port is None:
            self.port = {
                "sglang": 30000,
                "sglang-native": 30000,
                "sglang-oai": 30000,
                "lmdeploy": 23333,
                "vllm": 8000,
                "trt": 8000,
                "gserver": 9988,
                "truss": 8080,
            }.get(self.backend, 30000)

    def __post_init__(self):
        self._default_port()


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]
    # compatibility for openai apis
    encoding_format: str
    dimensions: Optional[int]


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""

    @staticmethod
    def init_new(request_func_input: RequestFuncInput):
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        return output


# trt llm does not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm_embedding(
    request_func_input: RequestFuncInput,
    args: BenchArgs,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError("async_request_trt_llm_embedding not implemented yet.")


# set ignore_eos True by default
async def async_request_openai_embedding(
    request_func_input: RequestFuncInput,
    args: BenchArgs,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Makes a request to the OpenAI Chat Completions API.

    Args:
        request_func_input: Input parameters for the request.
        pbar: Optional tqdm progress bar to update.

    Returns:
        RequestFuncOutput: Output of the request, including generated text,
                           latency, TTFT, ITL, and success status.
    """
    api_url = request_func_input.api_url
    assert api_url.endswith("embeddings"), "OpenAI Completions API URL must end with 'embeddings'."

    prompt = request_func_input.prompt

    async with _create_bench_client_session() as session:
        payload = {
            "model": request_func_input.model,
            "input": prompt,
            "encoding_format": request_func_input.encoding_format,
            "dimensions": request_func_input.dimensions,
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput.init_new(request_func_input)

        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        latency = time.perf_counter() - st
                        data = json.loads(chunk_bytes)

                    output.success = True
                    output.latency = latency
                    output.output_len = sum(len(emb_obj["embedding"]) for emb_obj in data["data"])
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(f"{output.error=}")

    if pbar:
        pbar.update(1)
    return output


async def async_request_truss_embedding(
    request_func_input: RequestFuncInput,
    args: BenchArgs,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError("async_request_truss_embedding not implemented yet.")


async def async_request_sglang_embedding(
    request_func_input: RequestFuncInput,
    args: BenchArgs,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    prompt = request_func_input.prompt

    async with _create_bench_client_session() as session:
        payload = {
            ("text" if isinstance(prompt, str) else "input_ids"): prompt,
            **request_func_input.extra_request_body,
        }

        # Add image data if available (list of image urls/base64)
        if request_func_input.image_data:
            payload["image_data"] = request_func_input.image_data

        headers = get_auth_headers()

        output = RequestFuncOutput.init_new(request_func_input)

        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        latency = time.perf_counter() - st
                        data = json.loads(chunk_bytes)

                    output.success = True
                    output.latency = latency
                    output.output_len = len(data["embedding"])
                else:
                    output.error = response.reason or ""
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
    args: BenchArgs,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError()


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_sglang_embedding,
    "sglang-native": async_request_sglang_embedding,
    "sglang-oai": async_request_openai_embedding,
    "vllm": async_request_openai_embedding,
    "lmdeploy": async_request_openai_embedding,
    "trt": async_request_trt_llm_embedding,
    "gserver": async_request_gserver,
    "truss": async_request_truss_embedding,
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
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


def calculate_metrics(
    input_requests: List[DatasetRow],
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration on the benchmark arguments.",
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
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens)) / dur_s,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )

    return metrics, output_lens


async def benchmark(
    args: BenchArgs,
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    input_requests: List[DatasetRow],
    extra_request_body: Dict[str, Any],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, args=args, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, args=args, pbar=pbar)

    # Warmup
    print(f"Starting warmup with {args.warmup_requests} sequences...")

    # Use the first request for all warmup iterations
    test_request = input_requests[0]

    # Create the test input once
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=min(test_request.output_len, 32),
        image_data=test_request.image_data,
        extra_request_body=extra_request_body,
        encoding_format=args.encoding_format,
        dimensions=args.dimensions,
    )

    # Run warmup requests
    warmup_tasks = []
    for _ in range(args.warmup_requests):
        warmup_tasks.append(asyncio.create_task(request_func(request_func_input=test_input, args=args)))

    warmup_outputs: List[RequestFuncOutput] = await asyncio.gather(*warmup_tasks)

    # Check if at least one warmup request succeeded
    if args.warmup_requests > 0 and not any(output.success for output in warmup_outputs):
        raise ValueError(
            f"Warmup failed - Please make sure benchmark arguments are correctly specified. Error: {warmup_outputs[0].error}"
        )
    else:
        print(f"Warmup completed with {args.warmup_requests} sequences. Starting main benchmark run...")

    # Flush cache
    if ("sglang" in backend and _get_bool_env_var("SGLANG_IS_IN_CI")) or args.flush_cache:
        requests.post(base_url + "/flush_cache", headers=get_auth_headers())

    time.sleep(1.0)

    # Start profiler
    if args.profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/start_profile")
        if profile_output.success:
            print("Profiler started")

    pbar = None if args.disable_tqdm else tqdm(total=len(input_requests))

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, args.request_rate):
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
            image_data=request.image_data,
            extra_request_body=extra_request_body,
            encoding_format=args.encoding_format,
            dimensions=args.dimensions,
        )

        tasks.append(asyncio.create_task(limited_request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    # Stop profiler
    if args.profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    if "sglang" in backend:
        server_info = requests.get(base_url + "/get_server_info")
        if server_info.status_code == 200:
            server_info_json = server_info.json()
            if "decode" in server_info_json:
                server_info_json = server_info_json["decode"][0]
            accept_length = server_info_json["internal_states"][0].get("avg_spec_accept_length", None)
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
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", args.request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max request concurrency:",
            args.max_concurrency if args.max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated embeddings:", metrics.total_output))
    print("{:<40} {:<10}".format("Total generated embeddings (retokenized):", metrics.total_output_retokenized))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total token throughput (tok/s):", metrics.total_throughput))
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    if accept_length:
        print("{:<40} {:<10.2f}".format("Accept length:", accept_length))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", metrics.median_e2e_latency_ms))
    print("=" * 50)

    if metrics.mean_e2e_latency_ms is not None:
        result = {
            # Arguments
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            # Results
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_generaged_embeddings": metrics.total_output,
            "total_generaged_embeddings_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "concurrency": metrics.concurrency,
            "accept_length": accept_length,
        }
    else:
        print(f"Error running benchmark for request rate: {args.request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "random-image":
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_"
                f"{args.random_output_len}_{args.random_image_num_images}imgs_"
                f"{args.random_image_resolution}.jsonl"
            )
        elif args.dataset_name.startswith("random"):
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"

    result_details = {
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
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


def run_benchmark(args: BenchArgs):
    print(f"benchmark_args={args}")

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    if args.tokenize_prompt:
        assert args.backend == "sglang", "`--tokenize-prompt` only compatible with `--backend sglang` currently"

    # Set url
    model_url = f"{args.base_url}/v1/models" if args.base_url else f"http://{args.host}:{args.port}/v1/models"

    if args.backend in ["sglang", "sglang-native"]:
        api_url = f"{args.base_url}/encode" if args.base_url else f"http://{args.host}:{args.port}/encode"
    elif args.backend in ["sglang-oai", "vllm", "lmdeploy"]:
        api_url = f"{args.base_url}/v1/embeddings" if args.base_url else f"http://{args.host}:{args.port}/v1/embeddings"
    elif args.backend == "trt":
        # TODO: find tensorRT-llm's embedding model entrypoint
        api_url = (
            f"{args.base_url}/v2/models/ensemble/generate_stream"
            if args.base_url
            else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
        )
        if args.model is None:
            print("Please provide a model using `--model` when using `trt` backend.")
            sys.exit(1)
    elif args.backend == "gserver":
        # TODO: find gserver's embedding model entrypoint
        api_url = args.base_url if args.base_url else f"{args.host}:{args.port}"
        args.model = args.model or "default"
    elif args.backend == "truss":
        # TODO: find truss'es embedding model entrypoint
        api_url = (
            f"{args.base_url}/v1/models/model:predict"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/models/model:predict"
        )
    base_url = f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url

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
            print("Please specify the correct host and port using `--host` and `--port`.")
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
    input_requests = get_dataset(args, tokenizer)

    return asyncio.run(
        benchmark(
            args=args,
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            input_requests=input_requests,
            extra_request_body=extra_request_body,
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput for embedding models.")
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    bench_args = BenchArgs.from_cli_args(args)
    run_benchmark(bench_args)
