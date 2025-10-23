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


class ProfileLinks(BaseModel):
    """Pydantic model for profile trace links."""

    extend: Optional[str] = None
    decode: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Pydantic model for benchmark results table data, for a single isl and osl"""

    model_path: str
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    ttft: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_gen_throughput: float
    acc_length: Optional[float] = None
    profile_links: Optional[ProfileLinks] = None

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
        # Calculate costs (assuming H100 pricing for now)
        hourly_cost_per_gpu = 2  # $2/hour for one H100
        hourly_cost = hourly_cost_per_gpu * 1  # Assuming tp_size = 1 for simplicity
        input_util = 0.7
        accept_length = (
            round(self.acc_length, 2) if self.acc_length is not None else "n/a"
        )
        itl = 1 / (self.output_throughput / self.batch_size) * 1000
        input_cost = 1e6 / (self.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / self.output_throughput / 3600 * hourly_cost

        def get_perfetto_relay_link_from_trace_file(trace_file: str):
            import os
            from urllib.parse import quote

            rel_path = os.path.relpath(trace_file, trace_dir)
            raw_file_link = f"{base_url}/{rel_path}"
            relay_link = (
                f"{relay_base}?src={quote(raw_file_link, safe='')}"
                if relay_base and quote
                else raw_file_link
            )
            return relay_link

        # Handle profile links
        profile_link = "NA | NA"
        if self.profile_links:
            if self.profile_links.extend or self.profile_links.decode:
                # Create a combined link or use the first available one
                trace_files = [self.profile_links.extend, self.profile_links.decode]
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


def generate_markdown_report(trace_dir, results: List["BenchmarkResult"]) -> str:
    """Generate a markdown report from a list of BenchmarkResult object from a single run."""
    import os

    summary = f"### {results[0].model_path}\n"

    # summary += (
    #     f"Input lens: {result.input_len}. Output lens: {result.output_len}.\n"
    # )
    summary += "| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) | profile (extend) | profile (decode)|\n"
    summary += "| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ | --------------- | -------------- |\n"

    # all results should share the same isl & osl
    for result in results:
        base_url = os.getenv("TRACE_BASE_URL", "").rstrip("/")
        relay_base = os.getenv(
            "PERFETTO_RELAY_URL",
            "",
        ).rstrip("/")
        summary += result.to_markdown_row(trace_dir, base_url, relay_base)

    return summary


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    seed: int = 42
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_steps: int = 3
    profile_by_stage: bool = False
    profile_filename_prefix: str = None
    append_to_github_summary: bool = True
    dataset_path: str = ""
    parallel_batch: bool = False
    dataset_name: str = "random"
    output_path: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
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
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=["mmmu", "random"],
            help="Name of the dataset to benchmark on.",
        )
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
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
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
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument("--parallel-batch", action="store_true")
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
        )
        parser.add_argument(
            "--no-append-to-github-summary",
            action="store_false",
            dest="append_to_github_summary",
            help="Disable appending the output of this run to github ci summary",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=BenchArgs.output_path,
            help="Path to save benchmark results as JSON format. If not specified, results will only be saved to result-filename.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        kwargs = {}
        for attr, attr_type in attrs:
            val = getattr(args, attr)
            if attr_type is type(None):
                kwargs[attr] = val
            else:
                kwargs[attr] = attr_type(val)
        return cls(**kwargs)


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
    dataset_name="",
    profile: bool = False,
    profile_steps: int = 3,
    profile_by_stage: bool = False,
    profile_filename_prefix: str = None,
    dataset_path: str = "",
    parallel_batch: bool = False,
):
    requests.post(url + "/flush_cache")
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

    profile_link = None
    if profile:
        output_dir, profile_name = None, None
        if profile_filename_prefix:
            output_dir = os.path.dirname(profile_filename_prefix)
            profile_name = os.path.basename(profile_filename_prefix)
        profile_link: str = run_profile(
            url,
            profile_steps,
            ["CPU", "GPU"],
            output_dir,
            profile_name,
            profile_by_stage,
        )

    tic = time.perf_counter()

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

    response = requests.post(
        url + "/generate",
        json=payload,
        stream=True,
    )

    # The TTFT of the last request in the batch
    ttft = 0.0
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
                ttft = time.perf_counter() - tic

    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / ttft
    output_throughput = batch_size * output_len / (latency - ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency

    server_info = requests.get(url + "/get_server_info").json()
    acc_length = server_info["internal_states"][0].get("avg_spec_accept_length", None)
    last_gen_throughput = server_info["internal_states"][0]["last_gen_throughput"]

    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"ttft: {ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")

    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "latency": round(latency, 4),
                "output_throughput": round(output_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                "last_gen_throughput": round(last_gen_throughput, 2),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        profile_link,
    )


def save_results_as_json(result: List[Tuple], bench_args: BenchArgs, model: str):
    """Save benchmark results as JSON using Pydantic models."""
    json_results = []

    # Generate all parameter combinations to match with results
    param_combinations = list(
        itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        )
    )

    for i, (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        profile_link,
    ) in enumerate(result):
        # Get the corresponding parameters for this result
        bs, input_len, output_len = param_combinations[i]

        # Parse profile links if available
        profile_links = None
        if profile_link:
            profile_links = parse_profile_links(
                profile_link, batch_size, input_len, output_len
            )

        benchmark_result = BenchmarkResult(
            model_path=model,
            run_name=bench_args.run_name,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            latency=latency,
            ttft=ttft,
            input_throughput=input_throughput,
            output_throughput=output_throughput,
            overall_throughput=overall_throughput,
            last_gen_throughput=last_gen_throughput,
            acc_length=acc_length,
            profile_links=profile_links,
        )
        json_results.append(benchmark_result.model_dump())

    # Save to JSON file
    with open(bench_args.output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved as JSON to {bench_args.output_path}")


def parse_profile_links(
    profile_dir: str, batch_size: int, input_len: int, output_len: int
) -> Optional[ProfileLinks]:
    """Parse profile directory to extract extend and decode trace file links."""
    if not profile_dir or not os.path.exists(profile_dir):
        return None

    extend_link = None
    decode_link = None

    # Look for extend/prefill trace files
    for file in os.listdir(profile_dir):
        if file.endswith(".trace.json.gz") or file.endswith(".trace.json"):
            if "extend" in file.lower() or "prefill" in file.lower():
                extend_link = os.path.join(profile_dir, file)
            elif "decode" in file.lower():
                decode_link = os.path.join(profile_dir, file)

    # If no specific extend/decode files found, try to find files with batch/input/output info
    if not extend_link or not decode_link:
        for file in os.listdir(profile_dir):
            if file.endswith(".trace.json.gz") or file.endswith(".trace.json"):
                if f"_batch{batch_size}_input{input_len}_output{output_len}_" in file:
                    if "prefill" in file.lower() or "extend" in file.lower():
                        extend_link = os.path.join(profile_dir, file)
                    elif "decode" in file.lower():
                        decode_link = os.path.join(profile_dir, file)

    if extend_link or decode_link:
        return ProfileLinks(extend=extend_link, decode=decode_link)

    return None


def get_report_summary(
    result: List[Tuple], server_args: ServerArgs, bench_args: BenchArgs
):
    import tabulate

    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}.\n"
    )

    headers = [
        "batch size",
        "latency (s)",
        "input throughput (tok/s)",
        "output throughput (tok/s)",
        "acc length",
        "ITL (ms)",
        "input cost ($/1M)",
        "output cost ($/1M)",
    ]
    if bench_args.profile:
        headers.append("profile")
    rows = []

    for (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        _,
        _,
        acc_length,
        trace_link,
    ) in result:
        if is_blackwell():
            hourly_cost_per_gpu = 4  # $4/hour for one B200
        else:
            hourly_cost_per_gpu = 2  # $2/hour for one H100

        hourly_cost = hourly_cost_per_gpu * server_args.tp_size
        input_util = 0.7
        accept_length = round(acc_length, 2) if acc_length is not None else "n/a"
        itl = 1 / (output_throughput / batch_size) * 1000
        input_cost = 1e6 / (input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / output_throughput / 3600 * hourly_cost
        row = [
            batch_size,
            latency,
            input_throughput,
            output_throughput,
            accept_length,
            itl,
            input_cost,
            output_cost,
        ]
        if trace_link:
            row.append(f"[Profile]({trace_link})")
        rows.append(row)

    summary += tabulate.tabulate(
        rows, headers=headers, tablefmt="github", floatfmt=".2f"
    )
    return summary


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

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

    # warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        run_one_case(
            base_url,
            batch_size=16,
            input_len=1024,
            output_len=16,
            temperature=bench_args.temperature,
            return_logprob=bench_args.return_logprob,
            stream_interval=bench_args.client_stream_interval,
            input_len_step_percentage=bench_args.input_len_step_percentage,
            dataset_name=bench_args.dataset_name,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
            dataset_path=bench_args.dataset_path,
            parallel_batch=bench_args.parallel_batch,
        )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    # benchmark
    result = []
    bench_result = []
    try:
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            result.append(
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
                    dataset_name=bench_args.dataset_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
                    dataset_path=bench_args.dataset_path,
                    parallel_batch=bench_args.parallel_batch,
                    profile_filename_prefix=bench_args.profile_filename_prefix,
                )
            )

        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    bench_result.append(
                        (
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
                                profile=bench_args.profile,
                                profile_steps=bench_args.profile_steps,
                                profile_by_stage=bench_args.profile_by_stage,
                                dataset_path=bench_args.dataset_path,
                                parallel_batch=bench_args.parallel_batch,
                                profile_filename_prefix=bench_args.profile_filename_prefix,
                            )[-1],
                        )
                    )
                result = [t1[:-1] + t2 for t1, t2 in zip(result, bench_result)]
            except Exception as e:
                print(f"Error profiling, there will be no profile trace dump: {e}")
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    # Save results as JSON if output_path is specified
    if bench_args.output_path:
        save_results_as_json(result, bench_args, model=server_args.model_path)

    if not bench_args.show_report:
        return

    summary = get_report_summary(result, server_args, bench_args)

    if is_in_ci() and bench_args.append_to_github_summary:
        write_github_step_summary(summary)


def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)


if __name__ == "__main__":
    main()
