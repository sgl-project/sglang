"""
Benchmark the throughput of embedding models in offline (batch) mode.

Mirrors bench_offline_throughput.py but uses Engine.encode() instead of
Engine.generate(), so there are no sampling params or output-token metrics.

Usage
-----
# Random dataset (default)
python -m sglang.bench_offline_throughput_embedding \
    --model-path BAAI/bge-base-en-v1.5 \
    --num-prompts 1000

# ShareGPT prompts
python -m sglang.bench_offline_throughput_embedding \
    --model-path BAAI/bge-base-en-v1.5 \
    --dataset-name sharegpt \
    --num-prompts 1000

# Save results to a file
python -m sglang.bench_offline_throughput_embedding \
    --model-path BAAI/bge-base-en-v1.5 \
    --result-filename results.jsonl
"""

import argparse
import dataclasses
import json
import logging
import random
import time
from typing import Dict, List, Optional

import numpy as np

from sglang.benchmark.datasets import DatasetRow, get_dataset
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer, set_ulimit
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Datasets that cannot be used for embedding benchmarks (no text prompts).
_EMBEDDING_UNSUPPORTED_DATASETS = {"image", "mmmu", "mooncake"}


@dataclasses.dataclass
class BenchArgs:
    result_filename: str = ""
    dataset_name: str = "random"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 512
    random_range_ratio: float = 0.0
    seed: int = 1
    skip_warmup: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--result-filename",
            type=str,
            default=BenchArgs.result_filename,
            help="Append JSON result to this file.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=["random", "sharegpt"],
            help="Dataset to benchmark with.",
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default="",
            help="Path to the dataset file (required for sharegpt).",
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of embedding requests to run.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="Drop ShareGPT prompts longer than this many tokens.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Token length per request for the random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Fractional variation in request length for the random dataset.",
        )
        parser.add_argument("--seed", type=int, default=1, help="Random seed.")
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batch.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BenchArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def _make_dataset_args(bench_args: BenchArgs) -> argparse.Namespace:
    """Build a namespace compatible with get_dataset() from our BenchArgs."""
    return argparse.Namespace(
        dataset_name=bench_args.dataset_name,
        dataset_path=bench_args.dataset_path,
        num_prompts=bench_args.num_prompts,
        sharegpt_output_len=None,
        sharegpt_context_len=bench_args.sharegpt_context_len,
        random_input_len=bench_args.random_input_len,
        random_output_len=0,  # embeddings produce no output tokens
        random_range_ratio=bench_args.random_range_ratio,
        tokenize_prompt=False,
    )


def _encode_once(
    engine: Engine,
    reqs: List[DatasetRow],
) -> Dict:
    """Run a single encode pass and return timing + token stats."""
    prompts = [r.prompt for r in reqs]
    total_input_tokens = sum(r.prompt_len for r in reqs)

    st = time.perf_counter()
    enc_out = engine.encode(prompt=prompts)
    latency = time.perf_counter() - st

    # enc_out is List[Dict] for batch input, Dict for single input.
    if not isinstance(enc_out, list):
        enc_out = [enc_out]

    cached_tokens = sum(
        o.get("meta_info", {}).get("cached_tokens", 0) for o in enc_out
    )

    return {
        "successful_requests": len(reqs),
        "total_latency": latency,
        "total_input_tokens": total_input_tokens,
        "cached_tokens": cached_tokens,
        "request_throughput": len(reqs) / latency,
        "input_throughput": total_input_tokens / latency,
    }


def throughput_test(server_args: ServerArgs, bench_args: BenchArgs) -> Dict:
    if bench_args.dataset_name in _EMBEDDING_UNSUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{bench_args.dataset_name}' is not supported for embedding benchmarks. "
            f"Use one of: random, sharegpt."
        )

    engine = Engine(**dataclasses.asdict(server_args))
    try:
        tokenizer_id = server_args.tokenizer_path or server_args.model_path
        tokenizer = get_tokenizer(tokenizer_id)

        set_ulimit()
        random.seed(bench_args.seed)
        np.random.seed(bench_args.seed)

        dataset_args = _make_dataset_args(bench_args)
        input_requests = get_dataset(dataset_args, tokenizer)

        if not bench_args.skip_warmup:
            warmup_reqs = sample_random_requests(
                input_len=min(bench_args.random_input_len, 256),
                output_len=0,
                num_prompts=min(bench_args.num_prompts, 16),
                range_ratio=1.0,
                tokenizer=tokenizer,
                dataset_path=bench_args.dataset_path,
            )
            logger.info("Warmup...")
            _encode_once(engine, warmup_reqs)

        logger.info("Benchmark...")
        result = _encode_once(engine, input_requests)

        if bench_args.result_filename:
            with open(bench_args.result_filename, "a") as fout:
                fout.write(json.dumps(result) + "\n")

        _print_result(result)
        return result
    finally:
        engine.shutdown()


def _print_result(result: Dict) -> None:
    W = 50
    print("\n{s:{c}^{n}}".format(s=" Offline Embedding Throughput Result ", n=W, c="="))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print("{:<40} {:<10}".format("Cached tokens:", result["cached_tokens"]))
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
    print("=" * W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline embedding throughput benchmark for SGLang."
    )
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
