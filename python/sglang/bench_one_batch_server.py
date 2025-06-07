"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sglang.bench_one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --profile --profile-by-stage
"""

import argparse
import dataclasses
import itertools
import json
import multiprocessing
import os
import time
from typing import Tuple

import requests

from sglang.bench_serving import get_tokenizer, sample_random_requests
from sglang.profiler import run_profile
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import is_in_ci, write_github_step_summary


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    input_len_step_percentage: float = 0.0
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_by_stage: bool = False

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
        parser.add_argument("--profile-by-stage", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


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
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer,
    profile: bool = False,
    profile_by_stage: bool = False,
):
    requests.post(url + "/flush_cache")
    input_requests = sample_random_requests(
        input_len=input_len,
        output_len=output_len,
        num_prompts=batch_size,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path="",
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
        profile_link: str = run_profile(
            url, 3, ["CPU", "GPU"], None, None, profile_by_stage
        )

    tic = time.perf_counter()
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": [req.prompt for req in input_requests],
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "json_schema": json_schema,
            },
            "return_logprob": return_logprob,
            "stream": True,
        },
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
        profile_link if profile else None,
    )


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

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
            input_len_step_percentage=bench_args.input_len_step_percentage,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
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
                    input_len_step_percentage=bench_args.input_len_step_percentage,
                    run_name=bench_args.run_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
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
                                input_len_step_percentage=bench_args.input_len_step_percentage,
                                run_name=bench_args.run_name,
                                result_filename=bench_args.result_filename,
                                tokenizer=tokenizer,
                                profile=bench_args.profile,
                                profile_by_stage=bench_args.profile_by_stage,
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

    if not bench_args.show_report:
        return

    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}.\n"
    )
    summary += "| batch size | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |"

    if bench_args.profile:
        summary += " profile |"

    summary += "\n"
    summary += "| ---------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ |"

    if bench_args.profile:
        summary += "-------------|"
    summary += "\n"

    for (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        trace_link,
    ) in result:
        hourly_cost = 2 * server_args.tp_size  # $2/hour for one H100
        input_util = 0.7
        accept_length = round(acc_length, 2) if acc_length is not None else "n/a"
        line = (
            f"| {batch_size} | "
            f"{latency:.2f} | "
            f"{input_throughput:.2f} | "
            f"{output_throughput:.2f} | "
            f"{accept_length} | "
            f"{1 / (output_throughput/batch_size) * 1000:.2f} | "
            f"{1e6 / (input_throughput * input_util) / 3600 * hourly_cost:.2f} | "
            f"{1e6 / output_throughput / 3600 * hourly_cost:.2f} |"
        )
        if trace_link:
            line += f" [Profile]({trace_link}) |"
        line += "\n"
        summary += line

    # print metrics table
    print(summary)

    if is_in_ci():
        write_github_step_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)
