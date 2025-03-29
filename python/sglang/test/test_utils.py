"""Common utilities for testing and benchmarking"""

import argparse
import copy
import logging
import os
import random
import subprocess
import threading
import time
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F

from sglang.bench_serving import run_benchmark
from sglang.global_config import global_config
from sglang.lang.backend.openai import OpenAI
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.utils import get_bool_env_var, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.utils import get_exception_traceback

DEFAULT_FP8_MODEL_NAME_FOR_TEST = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
DEFAULT_FP8_MODEL_NAME_FOR_ACCURACY_TEST = "neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
DEFAULT_FP8_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST = (
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
)
DEFAULT_FP8_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST = (
    "nvidia/Llama-3.1-8B-Instruct-FP8"
)
# TODO(yundai424): right now specifying to an older revision since the latest one
#  carries kv cache quantization which doesn't work yet
DEFAULT_FP8_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST_REVISION = (
    "13858565416dbdc0b4e7a4a677fadfbd5b9e5bb9"
)

DEFAULT_MODEL_NAME_FOR_TEST = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MOE_MODEL_NAME_FOR_TEST = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST = "Qwen/Qwen1.5-MoE-A2.7B"
DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
DEFAULT_MLA_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST = "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"
DEFAULT_REASONING_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST = (
    "hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4"
)
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 1000
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 = "meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,google/gemma-2-27b-it"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2 = "meta-llama/Llama-3.1-70B-Instruct,mistralai/Mixtral-8x7B-Instruct-v0.1,Qwen/Qwen2-57B-A14B-Instruct"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1 = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8,neuralmagic/Mistral-7B-Instruct-v0.3-FP8,neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8,neuralmagic/gemma-2-2b-it-FP8"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2 = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8,neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8,neuralmagic/Qwen2-72B-Instruct-FP8,neuralmagic/Qwen2-57B-A14B-Instruct-FP8,neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_QUANT_TP1 = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4,hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4,hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SMALL_VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B"

DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST = "lmsys/sglang-EAGLE-llama2-chat-7B"

DEFAULT_IMAGE_URL = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
DEFAULT_VIDEO_URL = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4"


def is_in_ci():
    """Return whether it is in CI runner."""
    return get_bool_env_var("SGLANG_IS_IN_CI")


if is_in_ci():
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = 5157
    DEFAULT_URL_FOR_TEST = "http://127.0.0.1:6157"
else:
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = 1157
    DEFAULT_URL_FOR_TEST = "http://127.0.0.1:2157"


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


def add_common_sglang_args_and_parse(parser: argparse.ArgumentParser):
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--backend", type=str, default="srt")
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    args = parser.parse_args()
    return args


def select_sglang_backend(args: argparse.Namespace):
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
    elif args.backend == "gserver":
        return partial(call_generate_gserver, url=f"{args.host}:{args.port}")
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


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: Optional[str] = None,
    other_args: list[str] = (),
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    pd_seperated: bool = False,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    if pd_seperated:
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

    if pd_seperated:
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

    if api_key:
        command += ["--api-key", api_key]

    print(f"command={' '.join(command)}")

    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=return_stdout_stderr[0],
            stderr=return_stdout_stderr[1],
            env=env,
            text=True,
        )
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
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
                raise Exception(f"Server unexpectedly exits ({return_code=}).")

            time.sleep(10)

    kill_process_tree(process.pid)
    raise TimeoutError("Server failed to start within the timeout period.")


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
    tic = time.time()
    success = True

    for file in files:
        filename, estimated_time = file.name, file.estimated_time
        process = None

        def run_one_file(filename):
            nonlocal process

            filename = os.path.join(os.getcwd(), filename)
            print(f".\n.\nBegin:\npython3 {filename}\n.\n.\n", flush=True)
            tic = time.time()

            process = subprocess.Popen(
                ["python3", filename], stdout=None, stderr=None, env=os.environ
            )
            process.wait()
            elapsed = time.time() - tic

            print(
                f".\n.\nEnd:\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
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
        print(f"Success. Time elapsed: {time.time() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.time() - tic:.2f}s", flush=True)

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
    pd_seperated: bool = False,
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
        lora_name=None,
        prompt_suffix="",
        pd_seperated=pd_seperated,
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
):
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
    )

    try:
        if need_warmup:
            warmup_args = copy.deepcopy(args)
            warmup_args.num_prompts = 16
            run_benchmark(warmup_args)
        res = run_benchmark(args)
    finally:
        kill_process_tree(process.pid)

    assert res["completed"] == num_prompts
    return res


def run_bench_serving_multi(
    model,
    base_url,
    other_server_args,
    benchmark_args,
    need_warmup=False,
    pd_seperated=False,
):
    # Launch the server
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
        pd_seperated=pd_seperated,
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
    command = [
        "python3",
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        model,
        "--batch-size",
        "1",
        "--input",
        "128",
        "--output",
        "8",
        *[str(x) for x in other_args],
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        print(f"Output: {output}", flush=True)
        print(f"Error: {error}", flush=True)

        lastline = output.split("\n")[-3]
        output_throughput = float(lastline.split(" ")[-2])
    finally:
        kill_process_tree(process.pid)

    return output_throughput


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


STDERR_FILENAME = "stderr.txt"
STDOUT_FILENAME = "stdout.txt"


def read_output(output_lines: List[str], filename: str = STDERR_FILENAME):
    """Print the output in real time with another thread."""
    while not os.path.exists(filename):
        time.sleep(1)

    pt = 0
    while pt >= 0:
        if pt > 0 and not os.path.exists(filename):
            break
        lines = open(filename).readlines()
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


class CustomTestCase(unittest.TestCase):
    def _callTestMethod(self, method):
        _retry_execution(
            lambda: super(CustomTestCase, self)._callTestMethod(method),
            max_retry=_get_max_retry(),
        )


def _get_max_retry():
    return int(os.environ.get("SGLANG_TEST_MAX_RETRY", "2" if is_in_ci() else "0"))


def _retry_execution(fn, max_retry: int):
    if max_retry == 0:
        fn()
        return

    try:
        fn()
    except Exception as e:
        print(
            f"retry_execution failed once and will retry. This may be an error or a flaky test. Error: {e}"
        )
        traceback.print_exc()
        _retry_execution(fn, max_retry=max_retry - 1)
