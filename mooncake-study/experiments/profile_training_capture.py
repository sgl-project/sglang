"""Separate, annotated capture off/on traces. Never use these times as serving SLOs."""

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from benchmark_training_capture import (
    capture_config,
    capture_state,
    free_port,
    isolated_store,
    stop_process,
    wait_capture_idle,
    write_json,
)
from sglang.test import test_utils


def wait_reservations(url, count):
    deadline = time.monotonic() + 60
    while True:
        state = capture_state(url)
        if state is None or state["states"].get("available", 0) >= count:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Capture slots unavailable: {state}")
        time.sleep(0.1)


def generate(url, args, *, input_len, output_len, step):
    wait_reservations(url, args.concurrency)
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": [
                [1000 + step * args.concurrency + row] + [200] * (input_len - 1)
                for row in range(args.concurrency)
            ],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    outputs = response.json()
    if len(outputs) != args.concurrency or any(
        row["meta_info"]["completion_tokens"] != output_len for row in outputs
    ):
        raise RuntimeError("Profile probe did not complete the requested tokens")


def profile_workload(url, args, directory, *, input_len, output_len):
    for step in range(args.warmup_steps):
        generate(url, args, input_len=input_len, output_len=output_len, step=step)
    wait_capture_idle(url)
    requests.post(url + "/flush_cache", timeout=30).raise_for_status()
    before = capture_state(url)
    directory.mkdir()
    response = requests.post(
        url + "/start_profile",
        json={
            "output_dir": str(directory),
            "activities": ["CPU", "GPU"],
            "with_stack": True,
            "record_shapes": False,
            "profile_prefix": directory.name,
        },
        timeout=60,
    )
    response.raise_for_status()
    try:
        for step in range(args.steps):
            generate(
                url,
                args,
                input_len=input_len,
                output_len=output_len,
                step=args.warmup_steps + step,
            )
        after, _ = wait_capture_idle(url)
    finally:
        requests.post(url + "/stop_profile", timeout=180).raise_for_status()
    traces = list(directory.glob("*.trace.json*"))
    if not traces:
        raise RuntimeError("Profiler returned without a trace artifact")
    counters = (
        {
            key: value - before["counters"].get(key, 0)
            for key, value in after["counters"].items()
        }
        if after is not None
        else {}
    )
    if before is not None and counters.get("ready") != args.steps * args.concurrency:
        raise RuntimeError(f"Profile requests were not all captured: {counters}")
    result = {
        "input_len": input_len,
        "output_len": output_len,
        "batches": args.steps,
        "requests": args.steps * args.concurrency,
        "capture_counter_delta": counters,
        "traces": [str(path) for path in traces],
        "capture_after": after,
    }
    write_json(directory / "workload.json", result)
    return result


def profile_mode(args, directory, enabled):
    directory.mkdir()
    with isolated_store(directory, args.segment_mib << 20) as (
        address,
        _reader,
        catalog,
    ):
        url = f"http://127.0.0.1:{free_port()}"
        config = directory / "capture.json"
        server_args = [
            "--skip-server-warmup",
            "--skip-tokenizer-init",
            "--attention-backend",
            "triton",
            "--mem-fraction-static",
            "0.25",
            "--max-total-tokens",
            "4096",
            "--max-running-requests",
            str(args.concurrency),
            "--chunked-prefill-size",
            "128",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            str(args.concurrency),
        ]
        if enabled:
            write_json(config, capture_config(args, directory, address, catalog, 1.0))
            server_args += ["--training-capture-config", str(config)]
        launch = test_utils._launch_server_process

        def annotated(command, *launch_args):
            return launch(
                [sys.executable, "-m", "sglang.test.training_capture_profile_server"]
                + command[2:],
                *launch_args,
            )

        process = None
        try:
            with patch.object(test_utils, "_launch_server_process", annotated):
                process = test_utils.popen_launch_server(
                    args.model_path,
                    url,
                    timeout=300,
                    other_args=server_args,
                )
            return {
                "enabled": enabled,
                "prefill": profile_workload(
                    url,
                    args,
                    directory / "prefill",
                    input_len=args.input_len,
                    output_len=1,
                ),
                "decode": profile_workload(
                    url,
                    args,
                    directory / "decode",
                    input_len=1,
                    output_len=args.output_len,
                ),
            }
        finally:
            if process is not None:
                stop_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 14, 27])
    parser.add_argument("--kv-d2h-batch-tokens", type=int, default=1)
    parser.add_argument("--teacher-d2h-batch-tokens", type=int, default=1)
    parser.add_argument(
        "--teacher-topk-backend", choices=("torch", "flashinfer"), default="torch"
    )
    parser.add_argument("--device-mib", type=int, default=0)
    args = parser.parse_args()
    if (
        args.kv_d2h_batch_tokens < 1
        or args.teacher_d2h_batch_tokens < 1
        or args.device_mib < 0
        or (
            max(args.kv_d2h_batch_tokens, args.teacher_d2h_batch_tokens) > 1
            and not args.device_mib
        )
    ):
        parser.error("Batched KV D2H requires positive batching and device budget")
    if (
        min(args.input_len, args.output_len, args.concurrency, args.steps) < 1
        or args.warmup_steps < 0
    ):
        parser.error("Positive workload dimensions and nonnegative warmup are required")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.capture_slots, args.host_mib, args.segment_mib, args.seed = (
        max(16, args.concurrency),
        256,
        2048,
        42,
    )
    torch.set_num_threads(1)
    report = {
        "status": "running",
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "scope": "Profiler-only scopes; probes wait for capture slots. GPU work attribution, not serving latency or throughput.",
        "modes": [],
    }
    try:
        for enabled in (False, True):
            mode = profile_mode(
                args, args.output_dir / ("on" if enabled else "off"), enabled
            )
            report["modes"].append(mode)
            write_json(args.output_dir / "report.json", report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
