#!/usr/bin/env python3
"""Run a workload directly through the SGLang Simulator engine."""

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path


def configure_environment(
    sim_config: str, output_dir: str, mode: str, device: str
) -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(Path(sim_config).resolve())
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output)
    os.environ["SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH"] = str(
        output / "hicache_storage_keys.txt"
    )
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = mode
    if device == "cpu":
        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    else:
        os.environ.pop("SGLANG_USE_CPU_ENGINE", None)
    return output


def build_server_args(path: str, device: str, page_size: int | None):
    from sglang_simulator.compat import override_server_args

    from sglang.srt.server_args import ServerArgs

    with open(path, encoding="utf-8") as stream:
        raw = json.load(stream)
    if page_size is not None:
        raw["page_size"] = page_size
    kv_bytes = raw.pop("kv_bytes_per_token_per_gpu", None)
    for key in ("tp_size", "ep_size", "dp_size", "pp_size"):
        raw[key] = 1
    raw.update(
        {
            "load_format": "dummy",
            "device": device,
            "disable_cuda_graph": True,
            "attention_backend": "torch_native",
            "sampling_backend": "pytorch",
        }
    )
    allowed = set(inspect.signature(ServerArgs).parameters)
    args = ServerArgs(**{key: value for key, value in raw.items() if key in allowed})
    if kv_bytes is not None:
        override_server_args(args, kv_bytes_per_token_per_gpu=kv_bytes)
    return args


def persist(runner, metrics: dict, output: Path) -> None:
    (output / "result.metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    for name, rows in (
        ("result.request.jsonl", runner.get_request_stats()),
        ("result.iteration.jsonl", runner.get_iteration_stats()),
    ):
        with (output / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-args", required=True)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--mode", choices=["OFFLINE", "BLOCKING"], default="OFFLINE")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--page-size", type=int)
    parser.add_argument(
        "--workload", choices=["trace", "random", "sharegpt"], required=True
    )
    parser.add_argument("--dataset")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--timestamp-scale", type=float, default=1000.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = configure_environment(
        args.sim_config, args.output_dir, args.mode, args.device
    )

    from sglang_simulator.simulation.benchmark import BenchmarkConfig
    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner
    from sglang_simulator.workload import load_inprocess_workload

    server_args = build_server_args(args.server_args, args.device, args.page_size)
    with open(args.server_args, encoding="utf-8") as stream:
        raw_server_args = json.load(stream)
    dataset = load_inprocess_workload(
        name=args.workload,
        model_path=raw_server_args["model_path"],
        dataset_path=args.dataset,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        timestamp_scale=args.timestamp_scale,
    )
    benchmark = BenchmarkConfig(
        request_rate=args.request_rate,
        ignore_request_timestamp=args.workload != "trace",
    )

    shutil.copy(args.server_args, output / "server_args.json")
    shutil.copy(args.sim_config, output / "simulator.json")
    runner = SGLangBenchmarkRunner(server_args)
    try:
        metrics = runner.benchmark(benchmark, dataset)
        persist(runner, metrics, output)
        print(json.dumps(metrics, indent=2))
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
