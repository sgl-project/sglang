"""
Benchmark the throughput of embedding model in the offline mode.
It accepts server arguments (the same as launch_server.py) and benchmark arguments.

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_embedding_offline --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10

## Random dataset with default args
python -m sglang.bench_embedding_offline --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024
"""

import os
import time
import json
import asyncio
import logging
import inspect
import random
import argparse
import numpy as np
import dataclasses


from typing import Optional, List, Dict
from transformers import AutoConfig, AutoTokenizer

from sglang.bench_serving import (
    DatasetRow,
    get_dataset,
    get_tokenizer,
    sample_random_requests,
    set_ulimit,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.engine import Engine


@dataclasses.dataclass
class EmbeddingServerArgs(ServerArgs):
    def __post_init__(self):
        super().__post_init__()
        # always setup embedding model engine
        self.is_embedding = True


@dataclasses.dataclass
class BenchArgs:
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_output_len: Optional[int] = None
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    seed: int = 1
    apply_chat_template: bool = False
    profile: bool = False
    skip_warmup: bool = False
    do_not_exit: bool = False
    prompt_suffix: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename)
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument("--dataset-path", type=str, default="", help="Path to the dataset.")
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
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
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
            help="Range of sampled ratio of input/output length, used only for random dataset.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of groups with shared prefix, usedonly for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per group of shared prefix, usedonly for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="System prompt length, usedonly for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Question length, usedonly for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            help="Disable ignore EOS token",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--do-not-exit",
            action="store_true",
            help="Do not exit the program. This is useful for nsys profile with --duration and --delay.",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default="",
            help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def truncate_prompts(prompts: List[str], model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    max_length = getattr(config, "max_position_embeddings", 2048)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    truncated_prompts = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
        if len(tokens.input_ids[0]) > max_length:
            truncated_text = tokenizer.decode(tokens.input_ids[0][: max_length - 1], skip_special_tokens=True)
            truncated_prompts.append(truncated_text)
        else:
            truncated_prompts.append(prompt)
    return truncated_prompts


def throughput_test_once(
    backend: Engine,
    reqs: List[DatasetRow],
    model_path: str,
    profile: bool,
):
    measurement_results = {
        "backend": "engine",
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r.prompt_len for r in reqs),
        "average_e2e_latency": -1,
        "request_throughput": -1,
        "input_throughput": -1,
    }

    prompts = [r.prompt for r in reqs]
    prompts = truncate_prompts(prompts, model_path)
    image_data = [r.image_data for r in reqs] if reqs[0].image_data is not None else None

    if profile:
        assert "SGLANG_TORCH_PROFILER_DIR" in os.environ, "Please set SGLANG_TORCH_PROFILER_DIR."
        os.makedirs(os.environ["SGLANG_TORCH_PROFILER_DIR"], exist_ok=True)
        backend.start_profile()

    st = time.perf_counter()
    gen_out = backend.encode(prompt=prompts, image_data=image_data)
    latency = time.perf_counter() - st

    if profile:
        dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
        known_files = set(os.listdir(dir))
        backend.stop_profile()
        monitor_trace_file(known_files, dir)

    server_info = backend.get_server_info()

    measurement_results["total_latency"] = latency
    measurement_results["total_output"] = sum(len(o["embedding"]) for o in gen_out)
    measurement_results["average_e2e_latency"] = sum(o["meta_info"]["e2e_latency"] for o in gen_out) / len(gen_out)
    measurement_results["request_throughput"] = measurement_results["successful_requests"] / latency
    measurement_results["input_throughput"] = measurement_results["total_input_tokens"] / latency

    if inspect.isawaitable(server_info):
        server_info = asyncio.run(server_info)

    measurement_results["last_gen_throughput"] = server_info["internal_states"][0]["last_gen_throughput"]

    return measurement_results


def monitor_trace_file(known_files, directory, interval=1):
    print(f"Monitoring {directory} for new trace files...")

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    backend = Engine(**dataclasses.asdict(server_args))
    if not backend:
        raise ValueError("Please provide valid engine arguments")

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Set global environments
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Read dataset
    input_requests = get_dataset(bench_args, tokenizer)

    warmup_requests = sample_random_requests(
        input_len=256,
        output_len=16,
        num_prompts=min(bench_args.num_prompts, 16),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )

    # Warm up
    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend=backend,
            reqs=warmup_requests,
            model_path=server_args.model_path,
            profile=False,
        )
        time.sleep(0.5)

    logging.info("\nBenchmark...")
    result = throughput_test_once(
        backend=backend,
        reqs=input_requests,
        model_path=server_args.model_path,
        profile=bench_args.profile,
    )
    backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print("\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print("{:<40} {:<10}".format("Total generated embeddings:", result["total_output"]))
    print("{:<40} {:<10.2f}".format("Average e2e latency (s):", result["average_e2e_latency"]))
    print("{:<40} {:<10.2f}".format("Last generation throughput (tok/s):", result["last_gen_throughput"]))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", result["request_throughput"]))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", result["input_throughput"]))
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    # handling ModelScope model downloads
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() in ("true", "1"):
        if os.path.exists(args.model_path):
            print(f"Using local model path: {args.model_path}")
        else:
            try:
                from modelscope import snapshot_download

                print(f"Using ModelScope to download model: {args.model_path}")

                # download the model and replace args.model_path
                args.model_path = snapshot_download(
                    args.model_path,
                )
                print(f"Model downloaded to: {args.model_path}")
            except Exception as e:
                print(f"ModelScope download failed: {str(e)}")
                raise e

    server_args = EmbeddingServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)

    while bench_args.do_not_exit:
        pass
