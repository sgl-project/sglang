"""Common utilities for testing and benchmarking"""

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
import torch
import torch.nn.functional as F

from sglang.bench_serving import run_benchmark
from sglang.global_config import global_config
from sglang.srt.utils import (
    get_bool_env_var,
    get_device,
    is_port_available,
    kill_process_tree,
    retry,
)
from sglang.test.run_eval import run_eval
from sglang.utils import get_exception_traceback

# General test models
DEFAULT_MODEL_NAME_FOR_TEST = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE = "meta-llama/Llama-3.2-1B"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_MOE_MODEL_NAME_FOR_TEST = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE = "Qwen/Qwen1.5-MoE-A2.7B"
DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

# MLA test models
DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME_FOR_TEST = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_MLA_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST = "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"
DEFAULT_MODEL_NAME_FOR_TEST_MLA = "lmsys/sglang-ci-dsv3-test"
DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN = "lmsys/sglang-ci-dsv3-test-NextN"

# NVFP4 models
DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST = "nvidia/DeepSeek-V3-0324-FP4"

# FP8 models
DEFAULT_MODEL_NAME_FOR_TEST_FP8 = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
DEFAULT_MODEL_NAME_FOR_ACCURACY_TEST_FP8 = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
DEFAULT_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST_FP8 = (
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
)
DEFAULT_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST_FP8 = (
    "nvidia/Llama-3.1-8B-Instruct-FP8"
)
DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8 = "Qwen/Qwen3-1.7B-FP8"
DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE = "gaunernst/DeepSeek-V2-Lite-Chat-FP8"

# W8A8 models
DEFAULT_MODEL_NAME_FOR_TEST_W8A8 = "RedHatAI/Llama-3.2-3B-quantized.w8a8"
DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE = "nytopop/Qwen3-30B-A3B.w8a8"

# INT4 models
DEFAULT_MODEL_NAME_FOR_TEST_AWQ_INT4 = (
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)

# EAGLE
DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST = "lmsys/sglang-EAGLE-llama2-chat-7B"
DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3 = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EAGLE_DP_ATTENTION_TARGET_MODEL_FOR_TEST = "Qwen/Qwen3-30B-A3B"
DEFAULT_EAGLE_DP_ATTENTION_DRAFT_MODEL_FOR_TEST = "Tengyunw/qwen3_30b_moe_eagle3"
DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3 = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
DEFAULT_STANDALONE_SPECULATIVE_TARGET_MODEL_FOR_TEST = (
    "meta-llama/Llama-3.1-8B-Instruct"
)
DEFAULT_STANDALONE_SPECULATIVE_DRAFT_MODEL_FOR_TEST = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Other use cases
DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST = (
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  # auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  # auto_round:auto_awq
)
DEFAULT_MODEL_NAME_FOR_TEST_LOCAL_ATTENTION = (
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
DEFAULT_REASONING_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DEEPPEP_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-V3-0324"
DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST = (
    "hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4"
)
DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST = "Qwen/Qwen3-30B-A3B"
DEFAULT_DEEPSEEK_W4AFP8_MODEL_FOR_TEST = "Barrrrry/DeepSeek-R1-W4AFP8"

# Nightly tests
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 = "meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,google/gemma-2-27b-it"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2 = "meta-llama/Llama-3.1-70B-Instruct,mistralai/Mixtral-8x7B-Instruct-v0.1,Qwen/Qwen2-57B-A14B-Instruct"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1 = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8,neuralmagic/Mistral-7B-Instruct-v0.3-FP8,neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8,neuralmagic/gemma-2-2b-it-FP8"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2 = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8,neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8,neuralmagic/Qwen2-72B-Instruct-FP8,neuralmagic/Qwen2-57B-A14B-Instruct-FP8,neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8,zai-org/GLM-4.5-Air-FP8"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_QUANT_TP1 = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4,hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4,hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_IMAGE_URL = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
DEFAULT_VIDEO_URL = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4"

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600


def is_in_ci():
    """Return whether it is in CI runner."""
    return get_bool_env_var("SGLANG_IS_IN_CI")


def is_in_amd_ci():
    """Return whether it is in an AMD CI runner."""
    return get_bool_env_var("SGLANG_IS_IN_CI_AMD")


def is_blackwell_system():
    """Return whether it is running on a Blackwell (B200) system."""
    return get_bool_env_var("IS_BLACKWELL")


def _use_cached_default_models(model_repo: str):
    cache_dir = os.getenv("DEFAULT_MODEL_CACHE_DIR")
    if cache_dir and model_repo:
        model_path = os.path.join(cache_dir, model_repo)
        if os.path.isdir(model_path):
            return os.path.abspath(model_path)
    return ""


if is_in_ci():
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
        10000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 2000
    )
else:
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
        20000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 1000
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"

if is_in_amd_ci():
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3000

if is_blackwell_system():
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3000


def call_generate_lightllm(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop_sequences": stop,
        },
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    pred = res.json()["generated_text"][0]
    return pred


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def call_generate_vllm(prompt, temperature, max_tokens, stop=None, n=1, url=None):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    if n == 1:
        pred = res.json()["text"][0][len(prompt) :]
    else:
        pred = [x[len(prompt) :] for x in res.json()["text"]]
    return pred


def call_generate_outlines(
    prompt, temperature, max_tokens, stop=None, regex=None, n=1, url=None
):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "regex": regex,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    if n == 1:
        pred = res.json()["text"][0][len(prompt) :]
    else:
        pred = [x[len(prompt) :] for x in res.json()["text"]]
    return pred


def call_generate_srt_raw(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": stop,
        },
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    obj = res.json()
    pred = obj["text"]
    return pred


def call_generate_guidance(
    prompt, temperature, max_tokens, stop=None, n=1, regex=None, model=None
):
    assert model is not None
    from guidance import gen

    rets = []
    for _ in range(n):
        out = (
            model
            + prompt
            + gen(
                name="answer",
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                regex=regex,
            )
        )
        rets.append(out["answer"])
    return rets if n > 1 else rets[0]


def call_select_lightllm(context, choices, url=None):
    assert url is not None

    scores = []
    for i in range(len(choices)):
        data = {
            "inputs": context + choices[i],
            "parameters": {
                "max_new_tokens": 1,
            },
        }
        res = requests.post(url, json=data)
        assert res.status_code == 200
        scores.append(0)
    return np.argmax(scores)


def call_select_vllm(context, choices, url=None):
    assert url is not None

    scores = []
    for i in range(len(choices)):
        data = {
            "prompt": context + choices[i],
            "max_tokens": 1,
            "prompt_logprobs": 1,
        }
        res = requests.post(url, json=data)
        assert res.status_code == 200
        scores.append(res.json().get("prompt_score", 0))
    return np.argmax(scores)

    """
    Modify vllm/entrypoints/api_server.py

    if final_output.prompt_logprobs is not None:
        score = np.mean([prob[t_id] for t_id, prob in zip(final_output.prompt_token_ids[1:], final_output.prompt_logprobs[1:])])
        ret["prompt_score"] = score
    """


def call_select_guidance(context, choices, model=None):
    assert model is not None
    from guidance import select

    out = model + context + select(choices, name="answer")
    return choices.index(out["answer"])


def add_common_other_args_and_parse(parser: argparse.ArgumentParser):
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=[
            "vllm",
            "outlines",
            "lightllm",
            "gserver",
            "guidance",
            "srt-raw",
            "llama.cpp",
        ],
    )
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    args = parser.parse_args()

    if args.port is None:
        default_port = {
            "vllm": 21000,
            "outlines": 21000,
            "lightllm": 22000,
            "srt-raw": 30000,
            "gserver": 9988,
        }
        args.port = default_port.get(args.backend, None)
    return args


def auto_config_device() -> str:
    """Auto-config available device platform"""

    try:
        device = get_device()
    except (RuntimeError, ImportError) as e:
        print(f"Warning: {e} - Falling back to CPU")
        device = "cpu"

    return device


def add_common_sglang_args_and_parse(parser: argparse.ArgumentParser):
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--backend", type=str, default="srt")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "rocm", "cpu"],
        help="Device type (auto/cuda/rocm/cpu). Auto will detect available platforms",
    )
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--raw-result-file", type=str)
    args = parser.parse_args()

    return args


def select_sglang_backend(args: argparse.Namespace):
    from sglang.lang.backend.openai import OpenAI
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    if args.backend.startswith("srt"):
        if args.backend == "srt-no-parallel":
            global_config.enable_parallel_encoding = False
        backend = RuntimeEndpoint(f"{args.host}:{args.port}")
    elif args.backend.startswith("gpt-"):
        backend = OpenAI(args.backend)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    return backend


def _get_call_generate(args: argparse.Namespace):
    if args.backend == "lightllm":
        return partial(call_generate_lightllm, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "vllm":
        return partial(call_generate_vllm, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "srt-raw":
        return partial(call_generate_srt_raw, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "outlines":
        return partial(call_generate_outlines, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "guidance":
        from guidance import models

        model = models.LlamaCpp(args.model_path, n_gpu_layers=-1, n_ctx=args.n_ctx)
        call_generate = partial(call_generate_guidance, model=model)
        call_generate("Hello,", 1.0, 8, ".")
        return call_generate
    else:
        raise ValueError(f"Invalid backend: {args.backend}")


def _get_call_select(args: argparse.Namespace):
    if args.backend == "lightllm":
        return partial(call_select_lightllm, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "vllm":
        return partial(call_select_vllm, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "guidance":
        from guidance import models

        model = models.LlamaCpp(args.model_path, n_gpu_layers=-1, n_ctx=args.n_ctx)
        call_select = partial(call_select_guidance, model=model)

        call_select("Hello,", ["world", "earth"])
        return call_select
    else:
        raise ValueError(f"Invalid backend: {args.backend}")


def get_call_generate(args: argparse.Namespace):
    call_generate = _get_call_generate(args)

    def func(*args, **kwargs):
        try:
            return call_generate(*args, **kwargs)
        except Exception:
            print("Exception in call_generate:\n" + get_exception_traceback())
            raise

    return func


def get_call_select(args: argparse.Namespace):
    call_select = _get_call_select(args)

    def func(*args, **kwargs):
        try:
            return call_select(*args, **kwargs)
        except Exception:
            print("Exception in call_select:\n" + get_exception_traceback())
            raise

    return func


def _get_default_models():
    import inspect

    current_module = inspect.getmodule(_get_default_models)
    default_models = set()
    for name, value in current_module.__dict__.items():
        if (
            isinstance(name, str)
            and "DEFAULT_" in name
            and "MODEL_" in name
            and isinstance(value, str)
        ):
            if "," in value:
                parts = [part.strip() for part in value.split(",")]
                default_models.update(parts)
            else:
                default_models.add(value.strip())
    return json.dumps(list(default_models))


def try_cached_model(model_repo: str):
    model_dir = _use_cached_default_models(model_repo)
    return model_dir if model_dir else model_repo


def popen_with_error_check(command: list[str], allow_exit: bool = False):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run_and_check():
        stdout, stderr = process.communicate()

        while process.poll() is None:
            time.sleep(5)

        if not allow_exit or process.returncode != 0:
            raise Exception(
                f"{command} exited with code {process.returncode}\n{stdout=}\n{stderr=}"
            )

    t = threading.Thread(target=_run_and_check)
    t.start()
    return process


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    device: str = "auto",
    pd_separated: bool = False,
    num_replicas: Optional[int] = None,
):
    """Launch a server process with automatic device detection.

    Args:
        device: Device type ("auto", "cuda", "rocm" or "cpu").
                If "auto", will detect available platforms automatically.
    """
    other_args = other_args or []

    # Auto-detect device if needed
    if device == "auto":
        device = auto_config_device()
        other_args = list(other_args)
        other_args += ["--device", str(device)]

    _, host, port = base_url.split(":")
    host = host[2:]

    use_mixed_pd_engine = not pd_separated and num_replicas is not None
    if pd_separated or use_mixed_pd_engine:
        command = "sglang.launch_pd_server"
    else:
        command = "sglang.launch_server"

    command = [
        "python3",
        "-m",
        command,
        "--model-path",
        model,
        *[str(x) for x in other_args],
    ]

    if pd_separated or use_mixed_pd_engine:
        command.extend(
            [
                "--lb-host",
                host,
                "--lb-port",
                port,
            ]
        )
    else:
        command.extend(
            [
                "--host",
                host,
                "--port",
                port,
            ]
        )

    if use_mixed_pd_engine:
        command.extend(
            [
                "--mixed",
                "--num-replicas",
                str(num_replicas),
            ]
        )

    if api_key:
        command += ["--api-key", api_key]

    print(f"command={' '.join(command)}")

    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        def _dump(src, sinks):
            for line in iter(src.readline, ""):
                for sink in sinks:
                    sink.write(line)
                    sink.flush()
            src.close()

        threading.Thread(
            target=_dump,
            args=(process.stdout, [return_stdout_stderr[0], sys.stdout]),
            daemon=True,
        ).start()
        threading.Thread(
            target=_dump,
            args=(process.stderr, [return_stdout_stderr[1], sys.stderr]),
            daemon=True,
        ).start()
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    start_time = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:
            return_code = process.poll()
            if return_code is not None:
                # Server failed to start (non-zero exit code) or crashed
                raise Exception(
                    f"Server process exited with code {return_code}. "
                    "Check server logs for errors."
                )

            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health_generate",
                    headers=headers,
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass

            return_code = process.poll()
            if return_code is not None:
                raise Exception(
                    f"Server unexpectedly exits ({return_code=}). Usually there will be error logs describing the cause far above this line."
                )

            time.sleep(10)

    kill_process_tree(process.pid)
    raise TimeoutError("Server failed to start within the timeout period.")


def popen_launch_pd_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: Optional[str] = None,
    other_args: list[str] = (),
    env: Optional[dict] = None,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = "sglang.launch_server"

    command = [
        "python3",
        "-m",
        command,
        "--model-path",
        model,
        *[str(x) for x in other_args],
    ]

    command.extend(
        [
            "--host",
            host,
            "--port",
            port,
        ]
    )

    if api_key:
        command += ["--api-key", api_key]

    print(f"command={' '.join(command)}")

    process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    return process


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


def run_unittest_files(files: List[TestFile], timeout_per_file: float):
    tic = time.perf_counter()
    success = True

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time
        process = None

        def run_one_file(filename):
            nonlocal process

            filename = os.path.join(os.getcwd(), filename)
            print(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {filename}\n.\n.\n",
                flush=True,
            )
            tic = time.perf_counter()

            process = subprocess.Popen(
                ["python3", filename], stdout=None, stderr=None, env=os.environ
            )
            process.wait()
            elapsed = time.perf_counter() - tic

            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )
            return process.returncode

        try:
            ret_code = run_with_timeout(
                run_one_file, args=(filename,), timeout=timeout_per_file
            )
            assert (
                ret_code == 0
            ), f"expected return code 0, but {filename} returned {ret_code}"
        except TimeoutError:
            kill_process_tree(process.pid)
            time.sleep(5)
            print(
                f"\nTimeout after {timeout_per_file} seconds when running {filename}\n",
                flush=True,
            )
            success = False
            break

    if success:
        print(f"Success. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)

    return 0 if success else -1


def get_similarities(vec1, vec2):
    return F.cosine_similarity(torch.tensor(vec1), torch.tensor(vec2), dim=0)


def get_benchmark_args(
    base_url="",
    dataset_name="",
    dataset_path="",
    tokenizer="",
    num_prompts=500,
    sharegpt_output_len=None,
    random_input_len=4096,
    random_output_len=2048,
    sharegpt_context_len=None,
    request_rate=float("inf"),
    disable_stream=False,
    disable_ignore_eos=False,
    seed: int = 0,
    device="auto",
    pd_separated: bool = False,
    lora_name=None,
    lora_request_distribution="uniform",
    lora_zipf_alpha=1.5,
):
    return SimpleNamespace(
        backend="sglang",
        base_url=base_url,
        host=None,
        port=None,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        model=None,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        sharegpt_output_len=sharegpt_output_len,
        sharegpt_context_len=sharegpt_context_len,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        random_range_ratio=0.0,
        request_rate=request_rate,
        multi=None,
        output_file=None,
        disable_tqdm=False,
        disable_stream=disable_stream,
        return_logprob=False,
        seed=seed,
        disable_ignore_eos=disable_ignore_eos,
        extra_request_body=None,
        apply_chat_template=False,
        profile=None,
        lora_name=lora_name,
        lora_request_distribution=lora_request_distribution,
        lora_zipf_alpha=lora_zipf_alpha,
        prompt_suffix="",
        device=device,
        pd_separated=pd_separated,
    )


def run_bench_serving(
    model,
    num_prompts,
    request_rate,
    other_server_args,
    dataset_name="random",
    dataset_path="",
    tokenizer=None,
    random_input_len=4096,
    random_output_len=2048,
    sharegpt_context_len=None,
    disable_stream=False,
    disable_ignore_eos=False,
    need_warmup=False,
    seed: int = 0,
    device="auto",
    background_task: Optional[Callable[[str, asyncio.Event], Awaitable[None]]] = None,
    lora_name: Optional[str] = None,
):
    if device == "auto":
        device = auto_config_device()
    # Launch the server
    base_url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
    )

    # Run benchmark
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        sharegpt_context_len=sharegpt_context_len,
        request_rate=request_rate,
        disable_stream=disable_stream,
        disable_ignore_eos=disable_ignore_eos,
        seed=seed,
        device=device,
        lora_name=lora_name,
    )

    async def _run():
        if need_warmup:
            warmup_args = copy.deepcopy(args)
            warmup_args.num_prompts = 16
            await asyncio.to_thread(run_benchmark, warmup_args)

        start_event = asyncio.Event()
        stop_event = asyncio.Event()
        task_handle = (
            asyncio.create_task(background_task(base_url, start_event, stop_event))
            if background_task
            else None
        )

        try:
            start_event.set()
            result = await asyncio.to_thread(run_benchmark, args)
        finally:
            if task_handle:
                stop_event.set()
                await task_handle

        return result

    try:
        res = asyncio.run(_run())
    finally:
        kill_process_tree(process.pid)

    assert res["completed"] == num_prompts
    return res


def run_score_benchmark(
    model,
    num_requests=100,
    batch_size=5,
    other_server_args=None,
    need_warmup=False,
    device="auto",
):
    """Score API benchmark function compatible with run_bench_serving pattern"""
    if other_server_args is None:
        other_server_args = []

    if device == "auto":
        device = auto_config_device()

    # Launch the server (consistent with run_bench_serving)
    base_url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
    )

    async def _run_benchmark():

        # Load tokenizer for generating test data
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(model)

        # Score API configuration
        score_query_tokens = 120
        score_item_tokens = 180
        score_label_token_ids = [9454, 2753]  # Yes/No token IDs
        special_token = "<|im_start|>"

        def generate_text_with_token_count(num_tokens):
            """Generate text with precise token count using replicated token."""
            text = special_token * num_tokens
            actual_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if actual_tokens != num_tokens:
                text = special_token * (
                    num_tokens
                    // len(tokenizer.encode(special_token, add_special_tokens=False))
                )
            return text

        if need_warmup:
            warmup_data = {
                "query": generate_text_with_token_count(score_query_tokens),
                "items": [
                    generate_text_with_token_count(score_item_tokens) for _ in range(3)
                ],
                "label_token_ids": score_label_token_ids,
                "model": model,
                "apply_softmax": True,
            }

            async with aiohttp.ClientSession() as session:
                try:
                    await session.post(
                        f"{base_url}/v1/score",
                        json=warmup_data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    )
                except:
                    pass  # Ignore warmup errors

        test_requests = []
        for i in range(num_requests):
            query = generate_text_with_token_count(score_query_tokens)
            items = [
                generate_text_with_token_count(score_item_tokens)
                for _ in range(batch_size)
            ]

            score_data = {
                "query": query,
                "items": items,
                "label_token_ids": score_label_token_ids,
                "model": model,
                "apply_softmax": True,
            }
            test_requests.append(score_data)

        start_time = time.monotonic()
        successful_requests = 0
        total_latency = 0
        latencies = []

        async with aiohttp.ClientSession() as session:
            for request_data in test_requests:
                try:
                    request_start = time.monotonic()
                    async with session.post(
                        f"{base_url}/v1/score",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            request_end = time.monotonic()

                            if "scores" in response_data or "logprobs" in response_data:
                                latency_ms = (request_end - request_start) * 1000
                                latencies.append(latency_ms)
                                total_latency += latency_ms
                                successful_requests += 1
                except Exception:
                    continue

        end_time = time.monotonic()
        total_time = end_time - start_time

        if successful_requests > 0:
            throughput = successful_requests / total_time
            avg_latency = total_latency / successful_requests
            latencies.sort()
            p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0

            return {
                "completed": successful_requests,
                "total_requests": num_requests,
                "throughput": throughput,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "successful_requests": successful_requests,
            }
        else:
            return {
                "completed": 0,
                "total_requests": num_requests,
                "throughput": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "successful_requests": 0,
            }

    try:
        res = asyncio.run(_run_benchmark())
    finally:
        kill_process_tree(process.pid)

    assert res["completed"] == res["successful_requests"]
    return res


def run_bench_serving_multi(
    model,
    base_url,
    other_server_args,
    benchmark_args,
    need_warmup=False,
    pd_separated=False,
):
    # Launch the server
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
        pd_separated=pd_separated,
    )

    # run benchmark for all
    res_l = []
    try:
        for args in benchmark_args:
            if need_warmup:
                warmup_args = copy.deepcopy(args)
                warmup_args.num_prompts = 16
                run_benchmark(warmup_args)

            res = run_benchmark(args)
            res_l.append((args, res))
    finally:
        kill_process_tree(process.pid)

    return res_l


def run_bench_one_batch(model, other_args):
    """Launch a offline process with automatic device detection.

    Args:
        device: Device type ("auto", "cuda", "rocm" or "cpu").
                If "auto", will detect available platforms automatically.
    """
    # Auto-detect device if needed

    device = auto_config_device()
    print(f"Auto-configed device: {device}", flush=True)
    other_args += ["--device", str(device)]

    command = [
        "python3",
        "-m",
        "sglang.bench_one_batch",
        "--batch-size",
        "1",
        "--input",
        "128",
        "--output",
        "8",
        *[str(x) for x in other_args],
    ]
    if model is not None:
        command += ["--model-path", model]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        print(f"Output: {output}", flush=True)
        print(f"Error: {error}", flush=True)

        # Return prefill_latency, decode_throughput, decode_latency
        prefill_line = output.split("\n")[-9]
        decode_line = output.split("\n")[-3]
        pattern = (
            r"latency: (?P<latency>\d+\.\d+).*?throughput:\s*(?P<throughput>\d+\.\d+)"
        )
        match = re.search(pattern, prefill_line)
        if match:
            prefill_latency = float(match.group("latency"))
        match = re.search(pattern, decode_line)
        if match:
            decode_latency = float(match.group("latency"))
            decode_throughput = float(match.group("throughput"))
    finally:
        kill_process_tree(process.pid)

    return prefill_latency, decode_throughput, decode_latency


def run_bench_offline_throughput(model, other_args):
    command = [
        "python3",
        "-m",
        "sglang.bench_offline_throughput",
        "--num-prompts",
        "1",
        "--dataset-name",
        "random",
        "--random-input-len",
        "256",
        "--random-output-len",
        "256",
        "--model-path",
        model,
        *[str(x) for x in other_args],
    ]

    print(f"command={' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        print(f"Output: {output}", flush=True)
        print(f"Error: {error}", flush=True)

        output_throughput = -1
        for line in output.split("\n"):
            if "Last generation throughput (tok/s):" in line:
                output_throughput = float(line.split(":")[-1])
    finally:
        kill_process_tree(process.pid)

    return output_throughput


def run_bench_one_batch_server(
    model,
    base_url,
    server_args,
    bench_args,
    other_server_args,
    simulate_spec_acc_lens=None,
):
    from sglang.bench_one_batch_server import run_benchmark

    if simulate_spec_acc_lens is not None:
        env = {**os.environ, "SIMULATE_ACC_LEN": str(simulate_spec_acc_lens)}
    else:
        env = None

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
        env=env,
    )
    try:
        run_benchmark(server_args=server_args, bench_args=bench_args)
    finally:
        kill_process_tree(process.pid)


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def calculate_rouge_l(output_strs_list1, output_strs_list2):
    """calculate the ROUGE-L score"""
    rouge_l_scores = []

    for s1, s2 in zip(output_strs_list1, output_strs_list2):
        lcs_len = lcs(s1, s2)
        precision = lcs_len / len(s1) if len(s1) > 0 else 0
        recall = lcs_len / len(s2) if len(s2) > 0 else 0
        if precision + recall > 0:
            fmeasure = (2 * precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0
        rouge_l_scores.append(fmeasure)

    return rouge_l_scores


STDERR_FILENAME = "/tmp/stderr.txt"
STDOUT_FILENAME = "/tmp/stdout.txt"


def read_output(output_lines: List[str], filename: str = STDERR_FILENAME):
    """Print the output in real time with another thread."""
    while not os.path.exists(filename):
        time.sleep(0.01)

    pt = 0
    while pt >= 0:
        if pt > 0 and not os.path.exists(filename):
            break
        try:
            lines = open(filename).readlines()
        except FileNotFoundError:
            print(f"{pt=}, {os.path.exists(filename)=}")
            raise
        for line in lines[pt:]:
            print(line, end="", flush=True)
            output_lines.append(line)
            pt += 1
        time.sleep(0.1)


def run_and_check_memory_leak(
    workload_func,
    disable_radix_cache,
    enable_mixed_chunk,
    disable_overlap,
    chunked_prefill_size,
    assert_has_abort,
):
    other_args = [
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--log-level",
        "debug",
    ]
    if disable_radix_cache:
        other_args += ["--disable-radix-cache"]
    if enable_mixed_chunk:
        other_args += ["--enable-mixed-chunk"]
    if disable_overlap:
        other_args += ["--disable-overlap-schedule"]

    model = DEFAULT_MODEL_NAME_FOR_TEST
    port = random.randint(4000, 5000)
    base_url = f"http://127.0.0.1:{port}"

    # Create files and launch the server
    stdout = open(STDOUT_FILENAME, "w")
    stderr = open(STDERR_FILENAME, "w")
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        return_stdout_stderr=(stdout, stderr),
    )

    # Launch a thread to stream the output
    output_lines = []
    t = threading.Thread(target=read_output, args=(output_lines,))
    t.start()

    # Run the workload
    workload_func(base_url, model)

    # Clean up everything
    kill_process_tree(process.pid)
    stdout.close()
    stderr.close()
    if os.path.exists(STDOUT_FILENAME):
        os.remove(STDOUT_FILENAME)
    if os.path.exists(STDERR_FILENAME):
        os.remove(STDERR_FILENAME)
    kill_process_tree(process.pid)
    t.join()

    # Assert success
    has_new_server = False
    has_leak = False
    has_abort = False
    for line in output_lines:
        if "Uvicorn running" in line:
            has_new_server = True
        if "leak" in line:
            has_leak = True
        if "Abort" in line:
            has_abort = True

    assert has_new_server
    assert not has_leak
    if assert_has_abort:
        assert has_abort


def run_command_and_capture_output(command, env: Optional[dict] = None):
    stdout = open(STDOUT_FILENAME, "w")
    stderr = open(STDERR_FILENAME, "w")
    process = subprocess.Popen(
        command, stdout=stdout, stderr=stdout, env=env, text=True
    )

    # Launch a thread to stream the output
    output_lines = []
    t = threading.Thread(target=read_output, args=(output_lines, STDOUT_FILENAME))
    t.start()

    # Join the process
    process.wait()

    stdout.close()
    stderr.close()
    if os.path.exists(STDOUT_FILENAME):
        os.remove(STDOUT_FILENAME)
    if os.path.exists(STDERR_FILENAME):
        os.remove(STDERR_FILENAME)
    kill_process_tree(process.pid)
    t.join()

    return output_lines


def run_mmlu_test(
    disable_radix_cache=False,
    enable_mixed_chunk=False,
    disable_overlap=False,
    chunked_prefill_size=32,
):
    def workload_func(base_url, model):
        # Run the eval
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=128,
        )

        try:
            metrics = run_eval(args)
            assert metrics["score"] >= 0.65, f"{metrics=}"
        finally:
            pass

    run_and_check_memory_leak(
        workload_func,
        disable_radix_cache,
        enable_mixed_chunk,
        disable_overlap,
        chunked_prefill_size,
        assert_has_abort=False,
    )


def run_mulit_request_test(
    disable_radix_cache=False,
    enable_mixed_chunk=False,
    enable_overlap=False,
    chunked_prefill_size=32,
):
    def workload_func(base_url, model):
        def run_one(_):
            prompt = """
            System: You are a helpful assistant.
            User: What is the capital of France?
            Assistant: The capital of France is
            """

            response = requests.post(
                f"{base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                    },
                },
            )
            ret = response.json()

        with ThreadPoolExecutor(2) as executor:
            list(executor.map(run_one, list(range(4))))

    run_and_check_memory_leak(
        workload_func,
        disable_radix_cache,
        enable_mixed_chunk,
        enable_overlap,
        chunked_prefill_size,
        assert_has_abort=False,
    )


def write_github_step_summary(content):
    if not os.environ.get("GITHUB_STEP_SUMMARY"):
        logging.warning("GITHUB_STEP_SUMMARY environment variable not set")
        return

    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(content)


def run_logprob_check(self: unittest.TestCase, arg: Tuple):
    (
        input_len,
        output_len,
        temperature,
        logprob_start_len,
        return_logprob,
        top_logprobs_num,
    ) = arg
    input_ids = list(range(input_len))

    response = requests.post(
        self.base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
        },
    )
    response_json = response.json()

    res = response_json
    self.assertEqual(res["meta_info"]["prompt_tokens"], input_len)
    self.assertEqual(res["meta_info"]["completion_tokens"], output_len)

    # Test the number of tokens are correct
    if return_logprob:
        self.assertEqual(
            len(res["meta_info"]["input_token_logprobs"]) + logprob_start_len,
            res["meta_info"]["prompt_tokens"],
        )
        self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), output_len)

        if top_logprobs_num:
            self.assertEqual(
                len(res["meta_info"]["input_top_logprobs"]) + logprob_start_len,
                res["meta_info"]["prompt_tokens"],
            )
            self.assertEqual(len(res["meta_info"]["output_top_logprobs"]), output_len)

            for i in range(output_len):
                self.assertEqual(
                    len(res["meta_info"]["output_top_logprobs"][i]),
                    top_logprobs_num,
                )

                # Test the top-1 tokens are the same as output tokens if temperature == 0
                if temperature == 0:
                    rank = 0
                    while rank < len(res["meta_info"]["output_top_logprobs"][i]):
                        try:
                            self.assertListEqual(
                                res["meta_info"]["output_token_logprobs"][i],
                                res["meta_info"]["output_top_logprobs"][i][rank],
                            )
                            break
                        except AssertionError:
                            # There's a tie. Allow the second item in this case.
                            if (
                                res["meta_info"]["output_top_logprobs"][i][rank][0]
                                == res["meta_info"]["output_top_logprobs"][i][rank + 1][
                                    0
                                ]
                            ):
                                rank += 1
                            else:
                                raise


def send_generate_requests(base_url: str, num_requests: int) -> List[str]:
    """Sends generate request serially and returns status codes. Max concurrency is 1."""

    def generate():
        prompt = """
        System: You are a helpful assistant.
        User: What is the capital of France?
        Assistant: The capital of France is
        """
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 500,
                },
            },
        )
        return response.status_code

    return [generate() for _ in range(num_requests)]


async def send_concurrent_generate_requests(
    base_url: str, num_requests: int
) -> List[str]:
    """Sends generate request concurrently and returns status codes. Max concurrency is num_requests."""

    async def async_generate():
        async with aiohttp.ClientSession() as session:
            prompt = """
            System: You are a helpful assistant.
            User: What is the capital of France?
            Assistant: The capital of France is
            """
            async with session.post(
                f"{base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 500,
                    },
                },
            ) as response:
                return response.status

    tasks = [asyncio.create_task(async_generate()) for _ in range(num_requests)]
    return await asyncio.gather(*tasks)


async def send_concurrent_generate_requests_with_custom_params(
    base_url: str,
    custom_params: List[dict[str, Any]],
) -> Tuple[int, Any]:
    """Sends generate request concurrently with custom parameters and returns status code and response json tuple. Max concurrency is num_requests."""

    base_payload = {
        "text": """
                System: You are a helpful assistant.
                User: What is the capital of France?
                Assistant: The capital of France is
                """,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 500,
        },
    }

    async def async_generate_with_priority(req):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/generate",
                json=req,
            ) as response:
                resp_json = await response.json()
                return (response.status, resp_json)

    tasks = []
    for c in custom_params:
        req = base_payload.copy()
        req.update(c)
        tasks.append(asyncio.create_task(async_generate_with_priority(req)))
    return await asyncio.gather(*tasks)


class CustomTestCase(unittest.TestCase):
    def _callTestMethod(self, method):
        max_retry = int(
            os.environ.get("SGLANG_TEST_MAX_RETRY", "1" if is_in_ci() else "0")
        )
        retry(
            lambda: super(CustomTestCase, self)._callTestMethod(method),
            max_retry=max_retry,
        )

    def setUp(self):
        print(f"[Test Method] {self._testMethodName}", flush=True)


def dump_bench_raw_result(
    path: str,
    states,
    preds,
    labels,
):
    if not path:
        return

    rows = []
    for i in range(len(states)):
        state = states[i]
        output = state["answer"]
        prompt = _ensure_remove_suffix(state.text(), output)
        rows.append(
            dict(
                prompt_id=i,
                prompt=prompt,
                output=output,
                correct=bool(preds[i] == labels[i]),
            )
        )

    print(f"BenchRawResultDumper save results to {path}")
    Path(path).write_text("\n".join(json.dumps(row) for row in rows))


def _ensure_remove_suffix(text: str, suffix: str):
    assert text.endswith(suffix)
    return text.removesuffix(suffix)


class ModelLaunchSettings:
    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        extra_args: Optional[List[str]] = None,
        env: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.tp_size = tp_size
        self.extra_args = list(extra_args) if extra_args else []
        self.env = env

        if self.tp_size > 1 and "--tp" not in self.extra_args:
            self.extra_args.extend(["--tp", str(self.tp_size)])

        fixed_args = ["--enable-multimodal", "--trust-remote-code"]
        for fixed_arg in fixed_args:
            if fixed_arg not in self.extra_args:
                self.extra_args.append(fixed_arg)


class ModelEvalMetrics:
    def __init__(self, accuracy: float, eval_time: float):
        self.accuracy = accuracy
        self.eval_time = eval_time


def extract_trace_link_from_bench_one_batch_server_output(output: str) -> str:
    match = re.search(r"\[Profile\]\((.*?)\)", output)
    if match:
        trace_link = match.group(1)
        return trace_link
    return None


def parse_models(model_string: str):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def check_evaluation_test_results(
    results,
    test_name,
    model_accuracy_thresholds,
    model_latency_thresholds=None,
    model_count=None,
):
    """
    results: list of tuple of (model_path, accuracy, latency)
    """
    failed_models = []
    if model_latency_thresholds is not None:
        summary = " | model | status | score | score_threshold | latency | latency_threshold | \n"
        summary += "| ----- | ------ | ----- | --------------- | ------- | ----------------- | \n"
    else:
        summary = " | model | status | score | score_threshold | \n"
        summary += "| ----- | ------ | ----- | --------------- | \n"

    results_dict = {res[0]: (res[1], res[2]) for res in results}

    for model, accuracy_threshold in sorted(model_accuracy_thresholds.items()):
        latency_threshold = (
            model_latency_thresholds.get(model)
            if model_latency_thresholds is not None
            else 1e9
        )

        if model in results_dict:
            accuracy, latency = results_dict[model]
            is_success = accuracy >= accuracy_threshold and latency <= latency_threshold
            status_emoji = "✅" if is_success else "❌"

            if not is_success:
                if accuracy < accuracy_threshold:
                    failed_models.append(
                        f"\nScore Check Failed: {model}\n"
                        f"Model {model} score ({accuracy:.4f}) is below threshold ({accuracy_threshold:.4f})"
                    )
                if latency > latency_threshold:
                    failed_models.append(
                        f"\nLatency Check Failed: {model}\n"
                        f"Model {model} latency ({latency:.4f}) is above threshold ({latency_threshold:.4f})"
                    )

            if model_latency_thresholds is not None:
                line = f"| {model} | {status_emoji} | {accuracy} | {accuracy_threshold} | {latency} | {latency_threshold}\n"
            else:
                line = (
                    f"| {model} | {status_emoji} | {accuracy} | {accuracy_threshold}\n"
                )
        else:
            status_emoji = "❌"
            failed_models.append(f"Model failed to launch or be evaluated: {model}")
            if model_latency_thresholds is not None:
                line = f"| {model} | {status_emoji} | N/A | {accuracy_threshold} | N/A | {latency_threshold}\n"
            else:
                line = f"| {model} | {status_emoji} | N/A | {accuracy_threshold}\n"

        summary += line

    print(summary)

    if is_in_ci():
        write_github_step_summary(f"## {test_name}\n{summary}")

    if failed_models:
        print("Some models failed the evaluation.")
        raise AssertionError("\n".join(failed_models))


# Bench knobs for bench_one_batch_server (override by env)
def _parse_int_list_env(name: str, default_val: str):
    val = os.environ.get(name, default_val)
    return [int(x) for x in val.split(",") if x]


# Return filenames
def find_traces_under_path(path: str) -> List[str]:
    results = []
    for _, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".trace.json.gz"):
                results.append(f"{file}")
    return results


def write_results_to_json(model, metrics, mode="a"):
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "metrics": metrics,
        "score": metrics["score"],
    }

    if "latency" in metrics:
        result["latency"] = (metrics.get("latency"),)

    existing_results = []
    if mode == "a" and os.path.exists("results.json"):
        try:
            with open("results.json", "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    if isinstance(existing_results, list):
        existing_results.append(result)
    else:
        existing_results = [result]

    with open("results.json", "w") as f:
        json.dump(existing_results, f, indent=2)


def intel_amx_benchmark(extra_args=None, min_throughput=None):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self):
            common_args = [
                "--attention-backend",
                "intel_amx",
                "--disable-radix",
                "--trust-remote-code",
            ]
            full_args = common_args + (extra_args or [])

            model = test_func(self)
            prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
                model, full_args
            )

            print(f"{model=}")
            print(f"{prefill_latency=}")
            print(f"{decode_throughput=}")
            print(f"{decode_latency=}")

            if is_in_ci() and min_throughput is not None:
                self.assertGreater(decode_throughput, min_throughput)

        return wrapper

    return decorator
