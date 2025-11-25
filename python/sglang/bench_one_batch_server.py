"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sglang.bench_one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --profile --profile-by-stage
python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --output-path results.json --profile
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
from pydantic import BaseModel
from transformers import AutoProcessor, PreTrainedTokenizer

from sglang.bench_serving import (
    get_processor,
    get_tokenizer,
    sample_mmmu_requests,
    sample_random_requests,
)
from sglang.profiler import run_profile
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_blackwell, kill_process_tree
from sglang.test.test_utils import is_in_ci, write_github_step_summary

logger = logging.getLogger(__name__)


# TODO:
# There is huge redundancy between BenchmarkResult and BenchOneCaseResult, and redundancy between to_markdown_row, generate_markdown_report, get_report_summary.
# We should refactor them to reduce the code duplication.
# 1. Delete the BenchmarkResult use BenchOneCaseResult directly.
# 2. Merge all related markdown functions into BenchOneCaseResult


class BenchmarkResult(BaseModel):
    """Pydantic model for benchmark results table data, for a single isl and osl"""

    model_path: str
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_ttft: float
    last_gen_throughput: float
    acc_length: Optional[float] = None
    profile_link_extend: Optional[str] = None
    profile_link_decode: Optional[str] = None

    @staticmethod
    def help_str() -> str:
        return f"""
Note: To view the traces through perfetto-ui, please:
    1. open with Google Chrome
    2. allow popup
"""

    def to_markdown_row(
        self, trace_dir, base_url: str = "", relay_base: str = ""
    ) -> str:
        """Convert this benchmark result to a markdown table row."""

        hourly_cost_per_gpu = 2  # $2/hour for one H100
        hourly_cost = hourly_cost_per_gpu * 1  # Assuming tp_size = 1 for simplicity
        input_util = 0.7
        accept_length = round(self.acc_length, 2) if self.acc_length > 0 else "n/a"
        itl = 1 / (self.output_throughput / self.batch_size) * 1000
        input_cost = 1e6 / (self.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / self.output_throughput / 3600 * hourly_cost

        def get_perfetto_relay_link_from_trace_file(trace_file: str):
            from urllib.parse import quote

            rel_path = os.path.relpath(trace_file, trace_dir)
            raw_file_link = f"{base_url}/{rel_path}"
            relay_link = (
                f"{relay_base}?src={quote(raw_file_link, safe='')}"
                if relay_base
                else raw_file_link
            )
            return relay_link

        # Handle profile links
        profile_link = "NA | NA"
        if self.profile_link_extend or self.profile_link_decode:
            # Create a combined link or use the first available one
            trace_files = [self.profile_link_extend, self.profile_link_decode]
            if any(trace_file is None for trace_file in trace_files):
                logger.error("Some trace files are None", f"{trace_files=}")
            trace_files_relay_links = [
                (
                    f"[trace]({get_perfetto_relay_link_from_trace_file(trace_file)})"
                    if trace_file
                    else "N/A"
                )
                for trace_file in trace_files
            ]

            profile_link = " | ".join(trace_files_relay_links)

        # Build the row
        return f"| {self.batch_size} | {self.input_len} | {self.latency:.2f} | {self.input_throughput:.2f} | {self.output_throughput:.2f} | {accept_length} | {itl:.2f} | {input_cost:.2f} | {output_cost:.2f} | {profile_link} |\n"


def generate_markdown_report(trace_dir, results: List[BenchmarkResult]) -> str:
    """Generate a markdown report from a list of BenchmarkResult object from a single run."""
    # Build model header with run_name if it's not "default"
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    # Include GPU config in model header if available
    gpu_config = os.getenv("GPU_CONFIG", "")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"

    summary += "| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) | profile (extend) | profile (decode)|\n"
    summary += "| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ | ---------------- | --------------- |\n"

    # all results should share the same isl & osl
    for result in results:
        base_url = os.getenv("TRACE_BASE_URL", "").rstrip("/")
        relay_base = os.getenv(
            "PERFETTO_RELAY_URL",
            "",
        ).rstrip("/")
        summary += result.to_markdown_row(trace_dir, base_url, relay_base)

    return summary


def save_results_as_pydantic_models(
    results: List["BenchOneCaseResult"], pydantic_result_filename: str, model_path: str
):
    """Save benchmark results as JSON using Pydantic models."""
    json_results = []

    for res in results:
        profile_link_extend = None
        profile_link_decode = None

        if res.profile_link:
            for file in os.listdir(res.profile_link):
                if file.endswith(".trace.json.gz") or file.endswith(".trace.json"):
                    if "extend" in file.lower() or "prefill" in file.lower():
                        profile_link_extend = os.path.join(res.profile_link, file)
                    elif "decode" in file.lower():
                        profile_link_decode = os.path.join(res.profile_link, file)

        benchmark_result = BenchmarkResult(
            model_path=model_path,
            run_name=res.run_name,
            batch_size=res.batch_size,
            input_len=res.input_len,
            output_len=res.output_len,
            latency=res.latency,
            input_throughput=res.input_throughput,
            output_throughput=res.output_throughput,
            overall_throughput=res.overall_throughput,
            last_gen_throughput=res.last_gen_throughput,
            last_ttft=res.last_ttft,
            acc_length=res.acc_length,
            profile_link_extend=profile_link_extend,
            profile_link_decode=profile_link_decode,
        )
        json_results.append(benchmark_result.model_dump())

    with open(pydantic_result_filename, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_steps: int = 5
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    profile_output_dir: Optional[str] = None
    dataset_path: str = ""
    dataset_name: str = "random"
    parallel_batch: bool = False
    result_filename: str = "result.jsonl"
    pydantic_result_filename: Optional[str] = None
    append_to_github_summary: bool = True
    seed: int = 42

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix",
            type=str,
            default=BenchArgs.profile_prefix,
        )
        parser.add_argument(
            "--profile-output-dir",
            type=str,
            default=BenchArgs.profile_output_dir,
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=["mmmu", "random"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument("--parallel-batch", action="store_true")
        parser.add_argument(
            "--result-filename",
            type=str,
            default=BenchArgs.result_filename,
            help="Store the results line by line in the JSON Line format to this file.",
        )
        parser.add_argument(
            "--pydantic-result-filename",
            type=str,
            default=BenchArgs.pydantic_result_filename,
            help="Store the results as pydantic models in the JSON format to this file.",
        )
        parser.add_argument(
            "--no-append-to-github-summary",
            action="store_false",
            dest="append_to_github_summary",
            help="Disable appending the output of this run to github ci summary",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class BenchOneCaseResult(BaseModel):
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_ttft: float
    last_gen_throughput: float
    acc_length: float
    profile_link: Optional[str] = None

    def dump_to_jsonl(self, result_filename: str):
        with open(result_filename, "a") as fout:
            res = {
                "run_name": self.run_name,
                "batch_size": self.batch_size,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "latency": round(self.latency, 4),
                "input_throughput": round(self.input_throughput, 2),
                "output_throughput": round(self.output_throughput, 2),
                "overall_throughput": round(self.overall_throughput, 2),
                "last_ttft": round(self.last_ttft, 4),
                "last_gen_throughput": round(self.last_gen_throughput, 2),
                "acc_length": round(self.acc_length, 2),
            }
            fout.write(json.dumps(res) + "\n")


def launch_server_internal(server_args):
    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process(server_args: ServerArgs):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = 600

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer: PreTrainedTokenizer | AutoProcessor,
    profile: bool = False,
    profile_steps: int = BenchArgs.profile_steps,
    profile_by_stage: bool = False,
    profile_prefix: Optional[str] = BenchArgs.profile_prefix,
    profile_output_dir: Optional[str] = BenchArgs.profile_output_dir,
    dataset_name: str = BenchArgs.dataset_name,
    dataset_path: str = BenchArgs.dataset_path,
    parallel_batch: bool = False,
):
    requests.post(url + "/flush_cache")

    # Load input token ids
    # TODO: reuse bench_serving.get_dataset ?
    if dataset_name == "mmmu":
        input_requests = sample_mmmu_requests(
            num_requests=batch_size,
            processor=tokenizer,
            fixed_output_len=output_len,
            random_sample=False,
        )
    elif dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=batch_size,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            random_sample=True,
            return_text=False,
        )

    # Load sampling parameters
    use_structured_outputs = False
    if use_structured_outputs:
        texts = []
        for _ in range(batch_size):
            texts.append(
                "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                * 50
                + "Assistant:"
            )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    payload = {
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": output_len,
            "ignore_eos": True,
            "json_schema": json_schema,
            "stream_interval": stream_interval,
        },
        "return_logprob": return_logprob,
        "stream": True,
        **({"parallel_batch": parallel_batch} if parallel_batch else {}),
    }
    if dataset_name == "mmmu":
        # vlm
        input_ids = []
        # for vlms, tokenizer is an instance of AutoProcessor
        tokenizer = tokenizer.tokenizer
        for input_req in input_requests:
            input_ids += [tokenizer.encode(input_req.prompt)]
        payload["image_data"] = [req.image_data for req in input_requests]

    else:
        input_ids = [req.prompt for req in input_requests]

    payload["input_ids"] = input_ids

    # Turn on profiler
    profile_link = None
    if profile:
        profile_link: str = run_profile(
            url=url,
            num_steps=profile_steps,
            activities=["CPU", "GPU"],
            output_dir=profile_output_dir,
            profile_by_stage=profile_by_stage,
            profile_prefix=profile_prefix,
        )

    # Run the request
    tic = time.perf_counter()
    response = requests.post(
        url + "/generate",
        json=payload,
        stream=True,
    )

    # Get the TTFT of the last request in the batch
    last_ttft = 0.0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            if "error" in data:
                raise RuntimeError(f"Request has failed. {data}.")

            assert (
                data["meta_info"]["finish_reason"] is None
                or data["meta_info"]["finish_reason"]["type"] == "length"
            )
            if data["meta_info"]["completion_tokens"] == 1:
                last_ttft = time.perf_counter() - tic

    # Compute metrics
    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / last_ttft
    output_throughput = batch_size * output_len / (latency - last_ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency

    server_info = requests.get(url + "/get_server_info").json()
    internal_state = server_info.get("internal_states", [{}])
    last_gen_throughput = internal_state[0].get("last_gen_throughput", None) or -1
    acc_length = internal_state[0].get("avg_spec_accept_length", None) or -1

    # Print results
    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")
    print(f"last_ttft: {last_ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    if acc_length > 0:
        print(f"acc_length: {acc_length:.2f} ")

    # Dump results
    result = BenchOneCaseResult(
        run_name=run_name,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency=latency,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        overall_throughput=overall_throughput,
        last_ttft=last_ttft,
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        profile_link=profile_link,
    )

    # Save and return the results
    if result_filename:
        result.dump_to_jsonl(result_filename)

    return result


def should_skip_due_to_token_capacity(
    batch_size, input_len, output_len, skip_token_capacity_threshold
):
    if batch_size * (input_len + output_len) > skip_token_capacity_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} * ({input_len=} + {output_len=}) = {batch_size * (input_len + output_len)} > {skip_token_capacity_threshold=} due to kv cache limit."
            + "=" * 8
        )
        return True
    return False


def get_report_summary(
    results: List[BenchOneCaseResult], bench_args: BenchArgs, server_args: ServerArgs
):
    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}.\n"
    )
    summary += "| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |"

    if bench_args.profile:
        summary += " profile |"

    summary += "\n"
    summary += "| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ |"

    if bench_args.profile:
        summary += "-------------|"
    summary += "\n"

    if is_blackwell():
        hourly_cost_per_gpu = 4  # $4/hour for one B200
    else:
        hourly_cost_per_gpu = 2  # $2/hour for one H100
    input_util = 0.7

    # sort result by input_len
    results.sort(key=lambda x: x.input_len)
    for res in results:
        hourly_cost = hourly_cost_per_gpu * server_args.tp_size
        accept_length = round(res.acc_length, 2) if res.acc_length > 0 else "n/a"
        line = (
            f"| {res.batch_size} | "
            f"{res.input_len} | "
            f"{res.latency:.2f} | "
            f"{res.input_throughput:.2f} | "
            f"{res.output_throughput:.2f} | "
            f"{accept_length} | "
            f"{1 / (res.output_throughput/res.batch_size) * 1000:.2f} | "
            f"{1e6 / (res.input_throughput * input_util) / 3600 * hourly_cost:.2f} | "
            f"{1e6 / res.output_throughput / 3600 * hourly_cost:.2f} |"
        )
        if bench_args.profile:
            if res.profile_link:
                line += f" [Profile]({res.profile_link}) |"
            else:
                line += f" n/a |"
        line += "\n"
        summary += line

    return summary


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    # Get tokenizer
    server_info = requests.get(base_url + "/get_server_info").json()
    if "tokenizer_path" in server_info:
        tokenizer_path = server_info["tokenizer_path"]
    elif "prefill" in server_info:
        tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
    if bench_args.dataset_name == "mmmu":
        # mmmu implies this is a MLLM
        tokenizer = get_processor(tokenizer_path)
    else:
        tokenizer = get_tokenizer(tokenizer_path)

    # Get token capacity
    internal_state = server_info.get("internal_states", [{}])
    skip_token_capacity_threshold = (
        internal_state[0].get("memory_usage", {}).get("token_capacity", 1000000000)
    )

    # Warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        print(f"Warmup with batch_size={bench_args.batch_size}")
        for bs in bench_args.batch_size:
            run_one_case(
                base_url,
                batch_size=bs,
                input_len=1024,
                output_len=16,
                temperature=bench_args.temperature,
                return_logprob=bench_args.return_logprob,
                stream_interval=bench_args.client_stream_interval,
                input_len_step_percentage=bench_args.input_len_step_percentage,
                run_name="",
                result_filename="",
                tokenizer=tokenizer,
                dataset_name=bench_args.dataset_name,
                dataset_path=bench_args.dataset_path,
                parallel_batch=bench_args.parallel_batch,
            )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    results = []
    profile_results = []
    try:
        # Benchmark all cases
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            if should_skip_due_to_token_capacity(
                bs, il, ol, skip_token_capacity_threshold
            ):
                continue
            results.append(
                run_one_case(
                    base_url,
                    bs,
                    il,
                    ol,
                    temperature=bench_args.temperature,
                    return_logprob=bench_args.return_logprob,
                    stream_interval=bench_args.client_stream_interval,
                    input_len_step_percentage=bench_args.input_len_step_percentage,
                    run_name=bench_args.run_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
                    dataset_name=bench_args.dataset_name,
                    dataset_path=bench_args.dataset_path,
                    parallel_batch=bench_args.parallel_batch,
                )
            )

        # Profile all cases
        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    if should_skip_due_to_token_capacity(
                        bs, il, ol, skip_token_capacity_threshold
                    ):
                        continue
                    profile_prefix = (
                        bench_args.profile_prefix or ""
                    ) + f"-bs-{bs}-il-{il}"
                    profile_results.append(
                        run_one_case(
                            base_url,
                            bs,
                            il,
                            ol,
                            temperature=bench_args.temperature,
                            return_logprob=bench_args.return_logprob,
                            stream_interval=bench_args.client_stream_interval,
                            input_len_step_percentage=bench_args.input_len_step_percentage,
                            run_name=bench_args.run_name,
                            result_filename=bench_args.result_filename,
                            tokenizer=tokenizer,
                            dataset_name=bench_args.dataset_name,
                            dataset_path=bench_args.dataset_path,
                            parallel_batch=bench_args.parallel_batch,
                            profile=bench_args.profile,
                            profile_steps=bench_args.profile_steps,
                            profile_by_stage=bench_args.profile_by_stage,
                            profile_prefix=profile_prefix,
                            profile_output_dir=bench_args.profile_output_dir,
                        )
                    )

                # Replace the profile link
                for res, profile_res in zip(results, profile_results):
                    res.profile_link = profile_res.profile_link
            except Exception as e:
                print(f"Error profiling, there will be no profile trace dump: {e}")
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return

    # Print summary
    summary = get_report_summary(results, bench_args, server_args)
    print(summary)

    if is_in_ci() and bench_args.append_to_github_summary:
        write_github_step_summary(summary)

    # Save results as pydantic models in the JSON format
    if bench_args.pydantic_result_filename:
        save_results_as_pydantic_models(
            results,
            bench_args.pydantic_result_filename,
            model_path=server_args.model_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)
