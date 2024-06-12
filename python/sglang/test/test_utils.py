"""Common utilities for testing and benchmarking"""

import asyncio
from functools import partial

import numpy as np
import requests

from sglang.backend.openai import OpenAI
from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.global_config import global_config
from sglang.utils import get_exception_traceback


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
    prompt, temperature, max_tokens, stop=[], regex=None, n=1, url=None
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


def call_generate_ginfer(prompt, temperature, max_tokens, stop=None, url=None):
    import grpc
    from ginfer import sampler_pb2, sampler_pb2_grpc

    sampler_channel = grpc.insecure_channel(url.replace("http://", ""))
    sampler = sampler_pb2_grpc.SamplerStub(sampler_channel)

    if stop is None:
        stop_strings = None
    else:
        stop_strings = [stop]

    sample_request = sampler_pb2.SampleTextRequest(
        prompt=prompt,
        settings=sampler_pb2.SampleSettings(
            max_len=max_tokens,
            rng_seed=0,
            temperature=max(temperature, 1e-7),
            nucleus_p=1,
            stop_strings=stop_strings,
        ),
    )
    stream = sampler.SampleText(sample_request)
    response = "".join([x.text for x in stream])
    return response


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


async def call_generate_lmql(
    prompt, temperature, max_tokens, stop=None, n=1, max_len=4096, model=None, **kwargs
):
    assert model is not None
    import lmql

    if stop != None:

        @lmql.query(model=model)
        async def program(question, max_tokens, stop):
            '''lmql
            """{question}[ANSWER]""" where len(TOKENS(ANSWER)) < max_tokens and STOPS_AT(ANSWER, stop)
            return ANSWER
            '''

    else:

        @lmql.query(model=model)
        async def program(question, max_tokens):
            '''lmql
            """{question}[ANSWER]""" where len(TOKENS(ANSWER)) < max_tokens
            return ANSWER
            '''

    tasks = [
        program(
            question=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            max_len=max_len,
            **kwargs,
        )
        for _ in range(n)
    ]
    rets = await asyncio.gather(*tasks)
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


async def call_select_lmql(context, choices, temperature=0, max_len=4096, model=None):
    assert model is not None
    import lmql

    @lmql.query(model=model)
    async def program(ctx, choices):
        '''lmql
        """{ctx}[ANSWER]""" where ANSWER in set(choices)
        return ANSWER
        '''

    answer = await program(
        ctx=context, choices=choices, temperature=temperature, max_len=max_len
    )
    return choices.index(answer)


def add_common_other_args_and_parse(parser):
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
            "ginfer",
            "guidance",
            "lmql",
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
            "lmql": 23000,
            "srt-raw": 30000,
            "ginfer": 9988,
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


def _get_call_generate(args):
    if args.backend == "lightllm":
        return partial(call_generate_lightllm, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "vllm":
        return partial(call_generate_vllm, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "srt-raw":
        return partial(call_generate_srt_raw, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "ginfer":
        return partial(call_generate_ginfer, url=f"{args.host}:{args.port}")
    elif args.backend == "outlines":
        return partial(call_generate_outlines, url=f"{args.host}:{args.port}/generate")
    elif args.backend == "guidance":
        from guidance import models

        model = models.LlamaCpp(args.model_path, n_gpu_layers=-1, n_ctx=args.n_ctx)
        call_generate = partial(call_generate_guidance, model=model)
        call_generate("Hello,", 1.0, 8, ".")
        return call_generate
    elif args.backend == "lmql":
        import lmql

        model = lmql.model(args.model_path, endpoint=f"{args.host}:{args.port}")
        return partial(call_generate_lmql, model=model)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")


def _get_call_select(args):
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

    elif args.backend == "lmql":
        import lmql

        model = lmql.model(args.model_path, endpoint=f"{args.host}:{args.port}")
        return partial(call_select_lmql, model=model)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")


def get_call_generate(args):
    call_generate = _get_call_generate(args)

    def func(*args, **kwargs):
        try:
            return call_generate(*args, **kwargs)
        except Exception:
            print("Exception in call_generate:\n" + get_exception_traceback())
            raise

    return func


def get_call_select(args):
    call_select = _get_call_select(args)

    def func(*args, **kwargs):
        try:
            return call_select(*args, **kwargs)
        except Exception:
            print("Exception in call_select:\n" + get_exception_traceback())
            raise

    return func
