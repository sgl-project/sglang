"""
Benchmark the throughput of using the offline LLM engine.
This script does not launch a server.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (the same as bench_serving.py).

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct

## Random dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name random

## Shared prefix dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name generated-shared-prefix

## Sharegpt dataset on runtime backend
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --backend runtime
"""

import argparse
import dataclasses
import json
import logging
import random
import time
from typing import List, Optional, Tuple

import numpy as np

from sglang.api import Engine
from sglang.bench_serving import (
    get_dataset,
    get_tokenizer,
    sample_random_requests,
    set_ulimit,
)
from sglang.srt.server import Runtime
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_output_len: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    gen_num_groups: int = 64
    gen_prompts_per_group: int = 16
    gen_system_prompt_len: int = 2048
    gen_question_len: int = 128
    gen_output_len: int = 256
    disable_ignore_eos: bool = False
    seed: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--backend", type=str, default=BenchArgs.backend)
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
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
            help="Range of sampled ratio of input/output length, "
            "used only for random dataset.",
        )
        parser.add_argument(
            "--gen-num-groups",
            type=int,
            default=BenchArgs.gen_num_groups,
            help="Number of groups with shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gen-prompts-per-group",
            type=int,
            default=BenchArgs.gen_prompts_per_group,
            help="Number of prompts per group of shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gen-system-prompt-len",
            type=int,
            default=BenchArgs.gen_system_prompt_len,
            help="System prompt length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gen-question-len",
            type=int,
            default=BenchArgs.gen_question_len,
            help="Question length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gen-output-len",
            type=int,
            default=BenchArgs.gen_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )
        parser.add_argument(
            "--disable-ignore-eos",
            type=bool,
            default=BenchArgs.disable_ignore_eos,
            help="Disable ignore EOS token",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[Tuple[str, int, int]],
    ignore_eos: bool,
):
    measurement_results = {
        "backend": backend_name,
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r[1] for r in reqs),
        "total_output_tokens": -1,
        "request_throughput": -1,
        "input_throughput": -1,
        "output_throughput": -1,
        "total_throughput": -1,
    }

    prompt = [r[0] for r in reqs]
    sampling_params = [
        {
            "temperature": 0,
            "max_new_tokens": r[2],
            "ignore_eos": ignore_eos,
        }
        for r in reqs
    ]

    st = time.perf_counter()
    gen_out = backend.generate(prompt=prompt, sampling_params=sampling_params)
    latency = time.perf_counter() - st

    if backend_name == "runtime":
        gen_out = json.loads(gen_out)

    measurement_results["total_latency"] = latency
    measurement_results["total_output_tokens"] = sum(
        o["meta_info"]["completion_tokens"] for o in gen_out
    )
    measurement_results["request_throughput"] = (
        measurement_results["successful_requests"] / latency
    )
    measurement_results["input_throughput"] = (
        measurement_results["total_input_tokens"] / latency
    )
    measurement_results["output_throughput"] = (
        measurement_results["total_output_tokens"] / latency
    )
    measurement_results["total_throughput"] = (
        measurement_results["total_input_tokens"]
        + measurement_results["total_output_tokens"]
    ) / latency

    return measurement_results


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.backend == "engine":
        backend = Engine(**dataclasses.asdict(server_args))
        if not backend:
            raise ValueError("Please provide valid engine arguments")
    elif bench_args.backend == "runtime":
        backend = Runtime(**dataclasses.asdict(server_args))
    else:
        raise ValueError('Please set backend to either "engine" or "runtime"')

    tokenizer_id = server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Set global environmnets
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Read dataset
    input_requests = get_dataset(bench_args, tokenizer)

    warmup_requests = sample_random_requests(
        input_len=20,
        output_len=4,
        num_prompts=2,
        range_ratio=0.8,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )

    # Warm up
    throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=warmup_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
    )

    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
    )

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print(
        "\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="=")
    )
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"])
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result["input_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_throughput"]
        )
    )
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)
