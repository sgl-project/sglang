"""Capture on/off serving experiment with isolated real TCP Store and test Catalog.

Uses the existing streaming benchmark client. Snapshot validation runs after
measurement and producer exit. No target reference execution or observer hooks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

import requests
import torch
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    DTYPES,
    decode_manifest,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server
from sglang.test.training_capture_catalog import TestCaptureCatalog

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_process(process):
    if process.poll() is None:
        kill_process_tree(process.pid)
    process.wait(timeout=20)


def store_setup(address, segment_bytes=0):
    return {
        "local_hostname": f"127.0.0.1:{free_port()}",
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": segment_bytes,
        "local_buffer_size": 16 << 20,
        "protocol": "tcp",
        "rdma_devices": "",
        "master_server_addr": address,
    }


@contextmanager
def isolated_store(directory, segment_bytes):
    with ExitStack() as stack:
        log = stack.enter_context((directory / "master.log").open("w"))
        port = free_port()
        address = f"127.0.0.1:{port}"
        master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={port}",
                f"--metrics_port={free_port()}",
                f"--http_metadata_server_port={free_port()}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        stack.callback(stop_process, master)
        deadline = time.monotonic() + 30
        while True:
            if master.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"Store startup failed; see {directory}/master.log")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        segment = MooncakeSnapshotStore.connect(store_setup(address, segment_bytes))
        stack.callback(segment.close)
        reader = MooncakeSnapshotStore.connect(store_setup(address))
        stack.callback(reader.close)
        catalog = TestCaptureCatalog()
        stack.callback(catalog.close)
        yield address, reader, catalog


def capture_state(url):
    response = requests.get(url + "/server_info", timeout=10)
    response.raise_for_status()
    return response.json()["internal_states"][0].get("training_capture")


def wait_capture_idle(url, *, timeout=60):
    started = time.monotonic()
    while True:
        state = capture_state(url)
        if state is None or not any(
            count for name, count in state["states"].items() if name != "available"
        ):
            return state, time.monotonic() - started
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f"Capture did not drain: {state}")
        time.sleep(0.1)


def verify_stage_metrics(url, state, directory):
    """Check the actual multiprocess endpoint after client timing has stopped."""
    from prometheus_client.parser import text_string_to_metric_families

    stages = state.get("stage_timings", {}) if state else {}
    if not stages:
        return {"verified": False, "reason": "stage timings unavailable"}
    fields = {
        "stage_calls_total": "calls",
        "stage_failures_total": "errors",
        "stage_seconds_total": "seconds",
        "stage_max_seconds": "max_seconds",
    }
    deadline = time.monotonic() + 10
    path = directory / "metrics.prom"
    while True:
        response = requests.get(url + "/metrics", timeout=10)
        response.raise_for_status()
        observed = {stage: {} for stage in stages}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                suffix = sample.name.removeprefix("sglang:training_capture_")
                stage = sample.labels.get("stage")
                if suffix in fields and stage in observed:
                    field = fields[suffix]
                    if field in observed[stage]:
                        raise RuntimeError("Expected one producer per stage metric")
                    observed[stage][field] = sample.value
        if all(
            field in observed[stage]
            and math.isclose(observed[stage][field], value[field], rel_tol=1e-9)
            for stage, value in stages.items()
            for field in fields.values()
        ):
            path.write_text(response.text)
            return {"verified": True, "path": str(path), "observed": observed}
        if time.monotonic() >= deadline:
            path.write_text(response.text)
            raise RuntimeError(f"Stale or mismatched stage metrics; see {path}")
        time.sleep(0.1)


def warmup(url, args):
    for batch_size in (1, args.concurrency):
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": [
                    [1000 + row * 100 + i for i in range(args.input_len)]
                    for row in range(batch_size)
                ],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.output_len,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
    wait_capture_idle(url)
    response = requests.post(url + "/flush_cache", timeout=30)
    response.raise_for_status()


def capture_config(args, directory, address, catalog, ratio):
    return {
        "dataset_id": "capture-benchmark",
        "model_id": args.model_path,
        "producer_revision": args.source_revision,
        "selected_layer_ids": args.layers,
        "catalog_endpoint": catalog.endpoint,
        "journal_directory": str(directory / "journal"),
        "store": store_setup(address),
        "sample_ratio": ratio,
        "sample_seed": args.seed,
        "max_sample_tokens": args.input_len + args.output_len,
        "max_inflight_samples": args.capture_slots,
        "max_host_bytes": args.host_mib << 20,
        "kv_d2h_batch_tokens": args.kv_d2h_batch_tokens,
        "teacher_d2h_batch_tokens": args.teacher_d2h_batch_tokens,
        "max_device_bytes": args.device_mib << 20,
        "storage_chunk_tokens": 64,
    }


def benchmark_command(args, url, path):
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        url,
        "--model",
        args.model_path,
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "1.0",
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        "inf",
        "--seed",
        str(args.seed),
        "--warmup-requests",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--disable-tqdm",
        "--cache-report",
        "--output-file",
        str(path),
    ]


def measure(args, url, directory):
    before, _ = wait_capture_idle(url)
    command = benchmark_command(args, url, directory / "client.jsonl")
    with (directory / "client.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=args.phase_timeout,
        )
    rows = (directory / "client.jsonl").read_text().splitlines()
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one benchmark result")
    result = json.loads(rows[0])
    if result["completed"] != args.num_prompts:
        raise RuntimeError(f"Incomplete serving requests; see {directory}/client.log")
    if result["total_output_tokens"] != args.num_prompts * args.output_len:
        raise RuntimeError("Benchmark did not generate the configured token count")
    for metric in (
        "median_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "output_throughput",
    ):
        if not math.isfinite(result[metric]) or result[metric] <= 0:
            raise RuntimeError(
                f"Invalid streaming measurement {metric}={result[metric]}"
            )
    after, drain_seconds = wait_capture_idle(url)
    counters = (
        {
            key: value - before["counters"].get(key, 0)
            for key, value in after["counters"].items()
        }
        if after is not None
        else {}
    )
    metrics = {key: value for key, value in result.items() if key != "server_info"}
    metrics["request_rate"] = "inf"
    return {
        "client_command": command,
        "serving": metrics,
        "capture_before": before,
        "capture_after": after,
        "capture_counter_delta": counters,
        "stage_metrics": verify_stage_metrics(url, after, directory),
        "capture_stage_delta": (
            {
                stage: {
                    field: value[field]
                    - before.get("stage_timings", {}).get(stage, {}).get(field, 0)
                    for field in ("calls", "errors", "seconds")
                }
                for stage, value in after.get("stage_timings", {}).items()
            }
            if after is not None
            else {}
        ),
        "post_client_capture_drain_seconds": drain_seconds,
    }


def validate_publications(reader, publications, args):
    totals = Counter()
    for publication in publications:
        data = reader.get_tensor(
            publication["manifest_key"],
            [publication["manifest_nbytes"]],
            torch.uint8,
            publication["manifest_sha256"],
        )
        manifest = decode_manifest(bytes(tensor_bytes(data)))
        tensors = {
            obj.key: reader.get_tensor(
                obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256
            )
            for obj in manifest.objects
        }
        validate_tensors(manifest, tensors)
        if (manifest.sequence.prompt_length, manifest.sequence.response_length) != (
            args.input_len,
            args.output_len,
        ):
            raise RuntimeError("Snapshot length differs from the benchmark workload")
        totals["validated_samples"] += 1
        totals["manifest_bytes"] += publication["manifest_nbytes"]
        for obj in manifest.objects:
            totals["payload_bytes"] += obj.nbytes
            totals[f"{obj.kind}_payload_bytes"] += obj.nbytes
            totals["payload_objects"] += 1
    return dict(totals)


def run_phase(args, directory, ratio):
    directory.mkdir()
    with isolated_store(directory, args.segment_mib << 20) as (
        address,
        reader,
        catalog,
    ):
        config_path = directory / "capture.json"
        if ratio is not None:
            write_json(
                config_path, capture_config(args, directory, address, catalog, ratio)
            )
        url = f"http://127.0.0.1:{free_port()}"
        server_args = [
            "--skip-server-warmup",
            "--enable-metrics",
            "--enable-cache-report",
            "--attention-backend",
            "triton",
            "--mem-fraction-static",
            "0.25",
            "--max-total-tokens",
            str(max(4096, args.concurrency * (args.input_len + args.output_len) * 2)),
            "--max-running-requests",
            str(args.concurrency),
            "--chunked-prefill-size",
            "128",
            "--random-seed",
            str(args.seed),
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            str(args.concurrency),
        ]
        if ratio is not None:
            server_args += ["--training-capture-config", str(config_path)]
        process = None
        try:
            process = popen_launch_server(
                args.model_path, url, timeout=300, other_args=server_args
            )
            warmup(url, args)
            with catalog.condition:
                warmup_publications = set(catalog.publications)
            result = measure(args, url, directory)
            with catalog.condition:
                publications = [
                    value
                    for key, value in catalog.publications.items()
                    if key not in warmup_publications
                ]
                catalog_errors = list(catalog.errors)
            result.update(
                sample_ratio=ratio,
                server_args=server_args,
                catalog_errors=catalog_errors,
            )
            write_json(directory / "measurement.json", result)
        finally:
            if process is not None:
                stop_process(process)
        result["readback"] = validate_publications(reader, publications, args)
        admitted = result["capture_counter_delta"].get("admitted", 0)
        ready = result["capture_counter_delta"].get("ready", 0)
        if len(publications) != ready or catalog_errors:
            raise RuntimeError(
                "Catalog and producer disagree about completed snapshots"
            )
        if (
            ratio is not None
            and result["capture_counter_delta"].get("considered", 0) != args.num_prompts
        ):
            raise RuntimeError("Capture request accounting differs from the workload")
        selected = (
            args.num_prompts - result["capture_counter_delta"].get("sampled_out", 0)
            if ratio is not None
            else 0
        )
        result["selected_requests"] = selected
        result["actual_admission_fraction"] = admitted / args.num_prompts
        result["ready_fraction_of_admitted"] = ready / admitted if admitted else None
        result["ready_fraction_of_selected"] = ready / selected if selected else None
        result["sampling_evidence"] = "observed" if admitted else "no_captured_requests"
        write_json(directory / "measurement.json", result)
        return result


def compare(phases):
    baseline_metrics = (
        "median_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "output_throughput",
    )
    first, last = phases[0]["serving"], phases[-1]["serving"]
    baseline = {key: (first[key] + last[key]) / 2 for key in baseline_metrics}
    return {
        "baseline_bracket_mean": baseline,
        "baseline_after_over_before": {
            key: last[key] / first[key] for key in baseline_metrics
        },
        "capture_over_baseline": [
            {
                "sample_ratio": phase["sample_ratio"],
                "ratios": {
                    key: phase["serving"][key] / baseline[key]
                    for key in baseline_metrics
                },
            }
            for phase in phases[1:-1]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=2048)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 14, 27])
    parser.add_argument("--capture-slots", type=int, default=16)
    parser.add_argument("--host-mib", type=int, default=256)
    parser.add_argument("--kv-d2h-batch-tokens", type=int, default=1)
    parser.add_argument("--teacher-d2h-batch-tokens", type=int, default=1)
    parser.add_argument("--device-mib", type=int, default=0)
    parser.add_argument("--segment-mib", type=int, default=2048)
    parser.add_argument("--phase-timeout", type=int, default=600)
    args = parser.parse_args()
    if not Path(args.model_path).is_dir():
        parser.error("Use a local model directory")
    if (
        min(
            args.num_prompts,
            args.input_len,
            args.concurrency,
            args.repeats,
            args.capture_slots,
            args.host_mib,
            args.kv_d2h_batch_tokens,
            args.teacher_d2h_batch_tokens,
            args.segment_mib,
            args.phase_timeout,
        )
        < 1
        or args.output_len < 2
    ):
        parser.error(
            "Positive dimensions and at least two response tokens are required"
        )
    if any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in args.ratios):
        parser.error("Capture ratios must be finite values in [0, 1]")
    if args.device_mib < 0 or (
        max(args.kv_d2h_batch_tokens, args.teacher_d2h_batch_tokens) > 1
        and not args.device_mib
    ):
        parser.error("Batched D2H requires a positive --device-mib budget")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {
        "status": "running",
        "declared_source_revision": args.source_revision,
        "producer_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                (ROOT / "python/sglang/srt/training_capture").glob("*.py")
            )
        },
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "sglang-kernel",
                "mooncake-transfer-engine-cuda13",
            )
        },
        "scope": {
            "transport": "local TCP",
            "catalog": "HTTP test double",
            "adaptive": False,
            "overlap": True,
            "decode_cuda_graphs": True,
            "prefill_cuda_graphs": False,
            "client": "existing sglang.benchmark.serving, streaming native /generate",
            "workload": "seeded fixed-length random-ids, closed-loop concurrency, not production traffic",
            "timing": "Client timing excludes warmup, server launch, readback and post-response writer drain",
            "payload_bytes": "Validated manifest tensor sizes; not D2H or wire byte counters",
            "gpu_capture_kernel_time": None,
            "d2h_bytes": None,
            "rdma_bytes": None,
            "production_slo_pass": None,
        },
        "rounds": [],
    }
    output = args.output_dir / "report.json"
    write_json(output, report)
    try:
        for repetition in range(args.repeats):
            ratios = args.ratios if repetition % 2 == 0 else list(reversed(args.ratios))
            current = {"repetition": repetition, "phases": []}
            report["rounds"].append(current)
            for index, ratio in enumerate([None, *ratios, None]):
                label = "off" if ratio is None else str(ratio)
                directory = (
                    args.output_dir / f"round-{repetition}-phase-{index}-{label}"
                )
                result = run_phase(args, directory, ratio)
                current["phases"].append(result)
                write_json(output, report)
                print(
                    json.dumps(
                        {
                            "phase": directory.name,
                            "completed": result["serving"]["completed"],
                            "output_tokens_per_second": result["serving"][
                                "output_throughput"
                            ],
                            "capture": result["capture_counter_delta"],
                            "readback": result["readback"],
                        }
                    ),
                    flush=True,
                )
            current["comparison"] = compare(current["phases"])
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output, report)


if __name__ == "__main__":
    main()
