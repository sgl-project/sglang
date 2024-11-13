"""
Benchmark the throughput of using the offline LLM engine.
This script does not launch a server.
It accepts the same arguments as bench_latency.py

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3-8B-Instruct

## Random dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random

## Shared prefix dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3-8B-Instruct --dataset-name generated-shared-prefix

## Sharegpt dataset on runtime backend
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3-8B-Instruct --backend runtime
"""

import argparse
import dataclasses
import itertools
import json
import logging
import random
import time
from typing import Dict, List, Tuple, Union

import numpy as np

from sglang.api import Engine as getEngine
from sglang.bench_serving import (
    get_tokenizer,
    get_dataset,
    sample_generated_shared_prefix_requests,
    sample_random_requests,
    sample_sharegpt_requests,
    set_ulimit,
)
from sglang.srt.server import Engine, Runtime
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_output_len: int = 256
    random_input_len: int = 256
    random_output_len: int = 256
    random_range_ratio: float = 0.0
    gen_num_groups: int = 8
    gen_prompts_per_group: int = 16
    gen_system_prompt_len: int = 128
    gen_question_len: int = 256
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
            "--disable-ignore-eos",
            type=bool,
            default=BenchArgs.disable_ignore_eos,
            help="Disable ignore EOS token",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        print(attrs)
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def throughput_test_once(
    backend_name: str,
    backend: Union[Engine, Runtime],
    reqs: List[Tuple[str, int, int]],
    output_len: int,
    ignore_eos: bool,
):
    measurement_results = {
        "total_input_tokens": sum(r[1] for r in reqs),
    }

    prompt = [r[0] for r in reqs]
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": output_len,
        "ignore_eos": ignore_eos,
    }

    st = time.perf_counter()
    gen_out = backend.generate(prompt=prompt, sampling_params=sampling_params)
    latency = time.perf_counter() - st

    if backend_name == "runtime":
        gen_out = json.loads(gen_out)

    measurement_results["total_latency"] = latency
    measurement_results["total_output_tokens"] = sum(
        o["meta_info"]["completion_tokens"] for o in gen_out
    )
    measurement_results["throughput"] = (
        measurement_results["total_input_tokens"]
        + measurement_results["total_output_tokens"]
    ) / latency

    print(f"Throughput: {measurement_results['throughput']} tokens/s")
    return measurement_results


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.backend == "engine":
        backend = getEngine(**dataclasses.asdict(server_args))
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
        output_len=output_len,
        ignore_eos=not bench_args.disable_ignore_eos,
    )

    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        output_len=output_len,
        ignore_eos=not bench_args.disable_ignore_eos,
    )

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

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

    try:
        print(throughput_test(server_args, bench_args))
    except Exception as e:
        raise e
