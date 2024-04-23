"""Common utilities for testing and benchmarking"""

import numpy as np
import requests
from sglang.backend.openai import OpenAI
from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.global_config import global_config


def call_generate_lightllm(prompt, temperature, max_tokens, stop, url):
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


def call_generate_vllm(prompt, temperature, max_tokens, stop, url, n=1):
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
    prompt, temperature, max_tokens, url, stop=[], regex=None, n=1
):
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


def call_generate_srt_raw(prompt, temperature, max_tokens, stop, url):
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


def call_select_lightllm(context, choices, url):
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


def call_select_vllm(context, choices, url):
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


def add_common_other_args_and_parse(parser):
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["vllm", "lightllm", "guidance", "lmql", "srt-raw", "llama.cpp"],
    )
    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    args = parser.parse_args()

    if args.port is None:
        default_port = {
            "vllm": 21000,
            "lightllm": 22000,
            "lmql": 23000,
            "srt-raw": 30000,
        }
        args.port = default_port.get(args.backend, None)
    return args


def add_common_sglang_args_and_parse(parser):
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--backend", type=str, default="srt")
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    args = parser.parse_args()
    return args


def select_sglang_backend(args):
    if args.backend.startswith("srt"):
        if args.backend == "srt-no-parallel":
            global_config.enable_parallel_decoding = False
            global_config.enable_parallel_encoding = False
        backend = RuntimeEndpoint(f"{args.host}:{args.port}")
    elif args.backend.startswith("gpt-"):
        backend = OpenAI(args.backend)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    return backend
