import argparse
import asyncio
import json
import random
import sys
from types import SimpleNamespace

import click
import numpy as np
import requests

from sglang.benchmark import args as bench_args
from sglang.benchmark.backends import get_api_url, get_backend_client
from sglang.benchmark.datasets import get_dataset_loader
from sglang.benchmark.runner import BenchmarkRunner
from sglang.benchmark.utils import (
    check_chat_template,
    get_auth_headers,
    get_tokenizer,
    set_ulimit,
)


def do_benchmark(args: argparse.Namespace):
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

    tokenizer_id = args.tokenizer or args.model
    tokenizer = get_tokenizer(tokenizer_id)

    api_url, base_url = get_api_url(args)

    print(f"Loading dataset: {args.dataset_name}")
    dataset_loader = get_dataset_loader(args, tokenizer)
    input_requests = dataset_loader.load()
    if not input_requests:
        print(
            "Error: Dataset is empty. Check dataset path and arguments.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(args, "flush_cache"):
        args.flush_cache = False

    backend_client = get_backend_client(args)

    runner = BenchmarkRunner(
        args=args,
        backend_client=backend_client,
        dataset_loader=dataset_loader,
        tokenizer=tokenizer,
        input_requests=input_requests,
        api_url=api_url,
        base_url=base_url,
        extra_request_body=extra_request_body,
    )

    try:
        asyncio.run(runner.run())
    except RuntimeError as e:
        print(f"\nAn error occurred during benchmark execution: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(0)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@bench_args.composite_options(bench_args.serving_options)
@bench_args.composite_options(bench_args.dataset_options)
@bench_args.composite_options(bench_args.common_benchmark_options)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    print(f"Benchmark arguments:\n{args}")
    do_benchmark(args)


if __name__ == "__main__":
    main()
