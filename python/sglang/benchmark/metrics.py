import json
import warnings
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import DatasetRow, RequestFuncOutput


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
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    accept_length: Optional[float] = None


def do_calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += outputs[i].prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
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
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )

    return metrics, output_lens


def print_metrics(metrics: BenchmarkMetrics, args: Namespace, duration: float):
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print(f"{'Backend:':<40} {args.backend:<10}")
    is_trace = args.dataset_name == "mooncake" and args.use_trace_timestamps
    print(f"{'Traffic request rate:':<40} {'trace' if is_trace else args.request_rate}")
    print(f"{'Max request concurrency:':<40} {args.max_concurrency or 'not set'}")
    print(f"{'Successful requests:':<40} {metrics.completed}")
    print(f"{'Benchmark duration (s):':<40} {duration:.2f}")
    print(f"{'Total input tokens:':<40} {metrics.total_input}")
    print(f"{'Total generated tokens:':<40} {metrics.total_output}")
    print(
        f"{'Total generated tokens (retokenized):':<40} {metrics.total_output_retokenized}"
    )
    print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:.2f}")
    print(f"{'Input token throughput (tok/s):':<40} {metrics.input_throughput:.2f}")
    print(f"{'Output token throughput (tok/s):':<40} {metrics.output_throughput:.2f}")
    print(f"{'Total token throughput (tok/s):':<40} {metrics.total_throughput:.2f}")
    print(f"{'Concurrency:':<40} {metrics.concurrency:.2f}")
    if metrics.accept_length:
        print(f"{'Accept length:':<40} {metrics.accept_length:.2f}")

    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(f"{'Mean E2E Latency (ms):':<40} {metrics.mean_e2e_latency_ms:.2f}")
    print(f"{'Median E2E Latency (ms):':<40} {metrics.median_e2e_latency_ms:.2f}")
    print(f"{'P99 E2E Latency (ms):':<40} {metrics.p99_e2e_latency_ms:.2f}")

    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print(f"{'Mean TTFT (ms):':<40} {metrics.mean_ttft_ms:.2f}")
    print(f"{'Median TTFT (ms):':<40} {metrics.median_ttft_ms:.2f}")
    print(f"{'P99 TTFT (ms):':<40} {metrics.p99_ttft_ms:.2f}")

    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print(f"{'Mean ITL (ms):':<40} {metrics.mean_itl_ms:.2f}")
    print(f"{'Median ITL (ms):':<40} {metrics.median_itl_ms:.2f}")
    print(f"{'P95 ITL (ms):':<40} {metrics.p95_itl_ms:.2f}")
    print(f"{'P99 ITL (ms):':<40} {metrics.p99_itl_ms:.2f}")
    print(f"{'Max ITL (ms):':<40} {metrics.max_itl_ms:.2f}")

    print("=" * 50)


def save_results(
    metrics: BenchmarkMetrics,
    args: Namespace,
    duration: float,
    outputs: List[RequestFuncOutput],
    output_lens: List[int],
):
    result = {
        # Arguments
        "backend": args.backend,
        "dataset_name": args.dataset_name,
        "request_rate": "trace" if args.use_trace_timestamps else args.request_rate,
        "max_concurrency": args.max_concurrency,
        "sharegpt_output_len": args.sharegpt_output_len,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "random_range_ratio": args.random_range_ratio,
        # Results
        "duration": duration,
        **metrics.__dict__,
    }

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
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"
            )

    if args.output_details:
        result["details"] = {
            "input_lens": [o.prompt_len for o in outputs],
            "output_lens": output_lens,
            "ttfts": [o.ttft for o in outputs],
            "itls": [o.itl for o in outputs],
            "generated_texts": [o.generated_text for o in outputs],
            "errors": [o.error for o in outputs],
        }

    with open(output_file_name, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"Results saved to {output_file_name}")
