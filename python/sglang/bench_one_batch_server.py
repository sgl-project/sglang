"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sglang.bench_one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sglang.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
"""

import argparse
import dataclasses
import itertools
import json
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
import torch
import torch.multiprocessing as mp

from sglang.bench_serving import get_tokenizer, sample_random_requests
from sglang.srt import fine_grained_benchmark
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    profile: bool = False
    profile_activities: Tuple[str] = ("CUDA_PROFILER",)
    profile_with_stack: bool = False
    profile_record_shapes: bool = False
    profile_skip_cases: int = 0
    enable_expert_distribution_recorder: bool = False
    expert_distribution_recorder_dir: str = "/tmp"
    seed: int = 1

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
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=BenchArgs.profile_activities,
        )
        parser.add_argument("--profile-with-stack", action="store_true")
        parser.add_argument("--profile-record-shapes", action="store_true")
        parser.add_argument(
            "--profile-skip-cases", type=int, default=BenchArgs.profile_skip_cases
        )
        parser.add_argument(
            "--enable-expert-distribution-recorder",
            action="store_true",
            help="Enable expert distribution recorder",
        )
        parser.add_argument(
            "--expert-distribution-recorder-dir",
            type=str,
            default=BenchArgs.expert_distribution_recorder_dir,
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")

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
    if fine_grained_benchmark.is_enabled():
        os.environ["SGLANG_ENABLE_COLOCATED_BATCH_GEN"] = "true"

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
    run_name: str,
    result_filename: str,
    tokenizer,
):
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

    if fine_grained_benchmark.is_enabled():
        fine_grained_benchmark.clear_output()

    tic = time.time()
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": [input_ids for input_ids, _, _ in input_requests],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
        },
    )
    latency = time.time() - tic

    _ = response.json()

    output_throughput = batch_size * output_len / latency
    overall_throughput = batch_size * (input_len + output_len) / latency

    print(f"batch size: {batch_size}")
    print(f"latency: {latency:.2f} s")
    print(f"output throughput: {output_throughput:.2f} token/s")
    print(f"(input + output) throughput: {overall_throughput:.2f} token/s")

    if fine_grained_benchmark.is_enabled():
        import pandas as pd

        fine_grained_output = fine_grained_benchmark.read_output()
        df = pd.DataFrame(fine_grained_output)
        df["throughput"] = df["num_tokens"] / df["latency"]
        with pd.option_context(
            "display.max_rows",
            10000,
            "display.max_columns",
            10000,
            "display.width",
            10000,
        ):
            print(df[df["tp_rank"] == 0].drop(["start_time", "tp_rank"], axis=1))

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
            }
            if fine_grained_benchmark.is_enabled():
                res["fine_grained_output"] = fine_grained_output
            fout.write(json.dumps(res) + "\n")


def _process_expert_distribution_record(bench_args, response):
    response.raise_for_status()
    data = response.json()
    path = (
        Path(bench_args.expert_distribution_recorder_dir) / "expert_distribution.json"
    )
    print(f"Write expert_distribution_recorder information to {path}", flush=True)
    path.write_text(json.dumps(data))

    import polars as pl

    df = pl.read_json(path, infer_schema_length=1000000)
    df = df.with_row_index("temp_index")
    df = df.explode("physical_count")
    df = df.with_columns(
        layer_id=(pl.col("physical_count").cum_count() - 1).over("temp_index")
    )
    df = df.drop("temp_index")
    df = df.with_columns(
        total_num_tokens=pl.col("physical_count").list.sum(),
        max_expert_num_tokens=pl.col("physical_count").list.max(),
        min_expert_num_tokens=pl.col("physical_count").list.min(),
    )
    df = df.filter(pl.col("total_num_tokens") > 0)
    df = df.sort("forward_pass_id", "layer_id", "gatherer_key", "rank")
    df = df.select(
        "forward_pass_id",
        "layer_id",
        "gatherer_key",
        "rank",
        "total_num_tokens",
        "max_expert_num_tokens",
        "min_expert_num_tokens",
        "physical_count",
    )
    with pl.Config(
        fmt_str_lengths=1000,
        tbl_cols=-1,
        tbl_rows=-1,
        fmt_table_cell_list_len=1000,
        tbl_width_chars=-1,
    ):
        print(df, flush=True)


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # warmup
    if not bench_args.skip_warmup:
        run_one_case(
            base_url,
            batch_size=16,
            input_len=1024,
            output_len=16,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
        )

    # benchmark
    try:
        for index, (bs, il, ol) in enumerate(
            itertools.product(
                bench_args.batch_size, bench_args.input_len, bench_args.output_len
            )
        ):
            if (
                bench_args.enable_expert_distribution_recorder
                and index == bench_args.profile_skip_cases
            ):
                requests.post(
                    base_url + "/start_expert_distribution_record"
                ).raise_for_status()
            if bench_args.profile and index == bench_args.profile_skip_cases:
                # TODO extract to PR
                print("bench script call cudaProfilerStart")
                torch.cuda.cudart().cudaProfilerStart()
                # print("Execute start_profile")
                # requests.post(
                #     base_url + "/start_profile",
                #     json={
                #         "activities": bench_args.profile_activities,
                #         "with_stack": bench_args.profile_with_stack,
                #         "record_shapes": bench_args.profile_record_shapes,
                #     },
                # ).raise_for_status()
            run_one_case(
                base_url,
                bs,
                il,
                ol,
                bench_args.run_name,
                bench_args.result_filename,
                tokenizer,
            )
        if bench_args.enable_expert_distribution_recorder:
            requests.post(
                base_url + "/stop_expert_distribution_record"
            ).raise_for_status()
            _process_expert_distribution_record(
                bench_args, requests.post(base_url + "/dump_expert_distribution_record")
            )
        if bench_args.profile:
            # TODO extract to PR
            print("bench script call cudaProfilerStop")
            torch.cuda.cudart().cudaProfilerStop()
            # print("Execute stop_profile")
            # requests.post(base_url + "/stop_profile").raise_for_status()
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)
