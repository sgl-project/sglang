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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.backends import get_backend_client
from sglang.benchmark.datasets import get_dataset_loader
from sglang.benchmark.datasets.common import DatasetRow, RequestFuncOutput
from sglang.benchmark.datasets.mmmu import MMMULoader
from sglang.benchmark.datasets.random import RandomLoader
from sglang.benchmark.metrics import BenchmarkMetrics, do_calculate_metrics
from sglang.benchmark.runner import BenchmarkRunner
from sglang.benchmark.serving import do_benchmark, main


# For compatibility
def run_benchmark(args_: argparse.Namespace):
    do_benchmark(args_)


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
) -> List[DatasetRow]:
    loader_args = argparse.Namespace()
    loader_args.random_input_len = input_len
    loader_args.random_output_len = output_len
    loader_args.num_prompts = num_prompts
    loader_args.random_range_ratio = range_ratio
    loader_args.dataset_path = dataset_path
    loader_args.dataset_name = "random" if random_sample else "random-ids"
    loader_args.tokenize_prompt = not return_text

    loader = RandomLoader(args=loader_args, tokenizer=tokenizer)

    return loader.load()


def sample_mmmu_requests(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    apply_chat_template: bool = True,
    random_sample: bool = True,
) -> List[DatasetRow]:
    loader_args = argparse.Namespace()
    loader_args.num_prompts = num_requests
    loader_args.random_output_len = fixed_output_len
    loader_args.apply_chat_template = apply_chat_template

    loader = MMMULoader(args=loader_args, tokenizer=tokenizer)

    return loader.load()


def get_dataset(args, tokenizer):
    dataset_loader = get_dataset_loader(args, tokenizer)
    return dataset_loader.load()


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
    extra_request_body: Dict[str, Any],
    profile: bool,
    pd_separated: bool = False,
    flush_cache: bool = False,
    warmup_requests: int = 1,
    use_trace_timestamps: bool = False,
    mooncake_slowdown_factor=1.0,
    mooncake_num_rounds=1,
):
    loader_args = argparse.Namespace()
    loader_args.backend = backend
    loader_args.model = model_id
    loader_args.request_rate = request_rate
    loader_args.max_concurrency = max_concurrency
    loader_args.disable_tqdm = disable_tqdm
    loader_args.lora_names = lora_names
    loader_args.profile = profile
    loader_args.pd_separated = pd_separated
    loader_args.flush_cache = flush_cache
    loader_args.warmup_requests = warmup_requests
    loader_args.use_trace_timestamps = use_trace_timestamps
    loader_args.mooncake_slowdown_factor = mooncake_slowdown_factor
    loader_args.mooncake_num_rounds = mooncake_num_rounds
    backend_client = get_backend_client(loader_args)
    dataset_loader = get_dataset_loader(loader_args, tokenizer)
    runner = BenchmarkRunner(
        args=loader_args,
        backend_client=backend_client,
        dataset_loader=dataset_loader,
        tokenizer=tokenizer,
        input_requests=input_requests,
        api_url=api_url,
        base_url=base_url,
        extra_request_body=extra_request_body,
    )

    return runner.run()


def calculate_metrics(
    input_requests: List[DatasetRow],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    return do_calculate_metrics(outputs, dur_s, tokenizer)


async def get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
) -> AsyncGenerator[DatasetRow, None]:
    input_requests_iter = iter(input_requests)
    for request in input_requests_iter:
        yield request

        if request_rate == float("inf"):
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


if __name__ == "__main__":
    main()
