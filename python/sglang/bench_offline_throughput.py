"""
Benchmark the throughput of using the offline LLM engine.
This script does not launch a server.
It accepts the same arguments as bench_latency.py

# Usage
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --result-filename out.jsonl

"""

import argparse
import dataclasses
import itertools
import logging
import time
import jsonlines
from typing import Dict, List, Tuple

from sglang.api import Engine as getEngine
from sglang.srt.server import Engine
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "before"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = ""
    # Plotting args
    graph_sql: str = (
        "select run_name, batch_size, prefill_throughput from results where run_name='before'"
    )
    graph_filename: str = "out.png"

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
        # graphing
        parser.add_argument("--graph-sql", type=str, default=BenchArgs.graph_sql)
        parser.add_argument(
            "--graph-filename", type=str, default=BenchArgs.graph_filename
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def prepare_synthetic_inputs_for_throughput_test(
    batch_size: int, input_len: int, output_len: int
):
    input_ids = [[1] * input_len for _ in range(batch_size)]
    sampling_params = {
        "temperature": 0,
        "min_new_tokens": output_len,
        "max_new_tokens": output_len,
    }
    return input_ids, sampling_params


def throughput_test_once(
    run_name: str,
    engine: Engine,
    reqs: Tuple[List[List[int]], Dict],
    output_len: int,
):
    measurement_results = {
        "run_name": run_name,
        "batch_size": len(reqs[0]),
        "input_len": len(reqs[0][0]),
        "output_len": output_len,
    }

    st = time.perf_counter()
    gen_out = engine.generate(input_ids=reqs[0], sampling_params=reqs[1])
    latency = time.perf_counter() - st

    measurement_results["total_latency"] = latency
    measurement_results["throughput"] = (
        (measurement_results["input_len"] + output_len)
        * measurement_results["batch_size"]
    ) / latency

    print(
        f"Throughput: BSZ {measurement_results['batch_size']} tokens, "
        f"Num sequences {len(reqs[0])}, throughput: "
        f"{measurement_results['throughput']} tokens/s"
    )
    return measurement_results


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    engine = getEngine(**dataclasses.asdict(server_args))
    if not engine:
        raise ValueError("Please provide valid engine arguments")

    warmup_reqs = prepare_synthetic_inputs_for_throughput_test(
        bench_args.batch_size[0], bench_args.input_len[0], bench_args.output_len[0]
    )

    # Warm up
    throughput_test_once("warmup", engine, warmup_reqs, bench_args.output_len[0])

    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        reqs = prepare_synthetic_inputs_for_throughput_test(bs, il, ol)
        ret = throughput_test_once(
            bench_args.run_name, engine, reqs, bench_args.output_len[0]
        )
        if ret is not None:
            result_list.append(ret)

    if bench_args.result_filename:
        with jsonlines.open(bench_args.result_filename, "a") as f:
            f.write_all(result_list)
    else:
        print(result_list)


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
        throughput_test(server_args, bench_args)
    except Exception as e:
        raise e
