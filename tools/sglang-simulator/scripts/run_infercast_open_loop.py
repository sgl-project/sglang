"""Replay a timestamped variable-length trace through SGLang Simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-total-tokens", type=int, default=262144)
    parser.add_argument("--max-prefill-tokens", type=int, default=196608)
    parser.add_argument("--chunked-prefill-size", type=int, default=196608)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--num-continuous-decode-steps", type=int, default=8)
    return parser.parse_args()


def load_trace(path: Path) -> SimpleDataset:
    requests = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            timestamp_ms = float(row["timestamp_ms"])
            input_length = int(row["input_length"])
            output_length = int(row["output_length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid trace row {line_number}: {error}") from error
        if timestamp_ms < 0 or input_length < 1 or output_length < 1:
            raise ValueError(f"invalid trace row {line_number}: values out of range")
        requests.append(
            GenericRequest(
                token_ids=[1000 + len(requests)] * input_length,
                input_length=input_length,
                output_length=output_length,
                custom_params={
                    "created_time": timestamp_ms / 1000.0,
                    "trace": {
                        key: value
                        for key, value in row.items()
                        if key not in {"timestamp_ms", "input_length", "output_length"}
                    },
                },
            )
        )
    if not requests:
        raise ValueError("trace must contain at least one request")
    return SimpleDataset(reqs=requests)


def main() -> None:
    args = parse_args()
    sim_config = args.sim_config.resolve()
    model_path = args.model_path.resolve()
    trace_path = args.trace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(sim_config)
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output_dir)

    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    dataset = load_trace(trace_path)
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(model_path),
            load_format="dummy",
            device="cpu",
            skip_tokenizer_init=True,
            max_total_tokens=args.max_total_tokens,
            max_prefill_tokens=args.max_prefill_tokens,
            chunked_prefill_size=args.chunked_prefill_size,
            mem_fraction_static=args.mem_fraction_static,
            num_continuous_decode_steps=args.num_continuous_decode_steps,
            page_size=256,
            disable_radix_cache=True,
        )
    )
    try:
        metrics = runner.benchmark(
            BenchmarkConfig(request_rate=float("inf"), ignore_request_timestamp=False),
            dataset=dataset,
        )
        if metrics is None:
            raise RuntimeError("SGLang Simulator did not produce metrics")
        result = {
            "method": "agentx_open_loop_diagnostic",
            "trace": str(trace_path),
            "request_count": len(dataset),
            "metrics": metrics,
        }
    finally:
        runner.shutdown()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
