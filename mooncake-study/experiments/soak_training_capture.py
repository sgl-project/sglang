"""Repeated real serving batches within one producer lifetime.

Uses the benchmark's real TCP Store, test Catalog and native streaming client.
Memory observations happen after each batch drains, without forced collection.
Every measured publication is read back only after the producer exits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import psutil
import torch
from benchmark_training_capture import (
    ROOT,
    apply_gc_policy,
    capture_config,
    diagnostic_server,
    free_port,
    gc_snapshot,
    isolated_store,
    measure,
    stop_process,
    validate_publications,
    verify_gc_lifecycle,
    wait_capture_idle,
    warmup,
    write_json,
)
from sglang.test.test_utils import popen_launch_server
from sglang.test.training_capture_benchmark_client import summarize_requests


def process_snapshot(pid):
    root = psutil.Process(pid)
    result = []
    for process in sorted([root, *root.children(recursive=True)], key=lambda p: p.pid):
        with process.oneshot():
            memory = process.memory_full_info()
            result.append(
                {
                    "pid": process.pid,
                    "created": process.create_time(),
                    "name": process.name(),
                    "command": process.cmdline(),
                    "rss_bytes": memory.rss,
                    "uss_bytes": memory.uss,
                    "pss_bytes": memory.pss,
                    "threads": process.num_threads(),
                    "fds": process.num_fds(),
                }
            )
    return result


def verify_drained(state, baseline, args, directory):
    if state is None:
        if baseline is not None:
            raise RuntimeError("Capture state disappeared")
        return
    if (
        state["disabled_reason"]
        or state["admission_paused"]
        or state["queued"]
        or any(count for name, count in state["states"].items() if name != "available")
        or state["reservations"] > args.capture_slots
    ):
        raise RuntimeError(f"Producer did not recover to idle: {state}")
    pool = state["host_pool"]
    if (
        pool["quarantined"]
        or pool["free"] + pool["filling"] != args.capture_slots
        or pool["allocated_bytes"] > args.host_mib << 20
        or pool["device_allocated_bytes"] > args.device_mib << 20
    ):
        raise RuntimeError(f"Capture pool escaped its ownership budget: {pool}")
    for field in ("allocated_bytes", "device_allocated_bytes"):
        if pool[field] != baseline["host_pool"][field]:
            raise RuntimeError(f"Capture pool grew: {field}")
    if list((directory / "journal").glob("*.json")):
        raise RuntimeError("Publication journal did not drain")


def memory_summary(observations, limit_mib):
    # First measured batch settles lazy allocations; every subsequent batch must
    # retain the same live processes. Report per-process growth, not summed RSS.
    settled = observations[1:]
    first = {item["pid"]: item for item in settled[0]["processes"]}
    for row in settled:
        current = {item["pid"]: item for item in row["processes"]}
        if current.keys() != first.keys() or any(
            current[pid]["created"] != item["created"] for pid, item in first.items()
        ):
            raise RuntimeError("Serving process lifetime changed during the soak")
    result = []
    for pid, base in first.items():
        values = [
            next(item for item in row["processes"] if item["pid"] == pid)
            for row in settled
        ]
        entry = {key: base[key] for key in ("pid", "created", "name", "command")}
        for field in ("rss_bytes", "uss_bytes", "pss_bytes", "threads", "fds"):
            entry[field] = {
                "first": base[field],
                "last": values[-1][field],
                "minimum": min(item[field] for item in values),
                "maximum": max(item[field] for item in values),
                "peak_growth": max(item[field] for item in values) - base[field],
            }
        entry["rss_within_budget"] = (
            entry["rss_bytes"]["peak_growth"] <= limit_mib << 20
        )
        result.append(entry)
    return result


def run(args):
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "config": vars(args) | {"output_dir": str(directory)},
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "sglang-kernel",
                "mooncake-transfer-engine-cuda13",
                "psutil",
            )
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [*Path(__file__).parent.glob("*training_capture.py")]
                + list((ROOT / "python/sglang/srt/training_capture").glob("*.py"))
                + [ROOT / "python/sglang/test/training_capture_pause_server.py"]
            )
        },
        "scope": {
            "catalog": "HTTP test double; no consumer or retention GC",
            "transport": "local TCP",
            "workload": "fixed lengths, distinct seeded random IDs per batch",
            "memory": "per-process RSS/USS/PSS at drained batch boundaries; not peak allocation or a leak proof",
            "collection": "no forced GC or cache flush between measured batches",
            "store": "retains all samples; segment/test Catalog live in the driver, outside serving process measurements",
            "production_slo_pass": None,
        },
        "batches": [],
        "observations": [],
    }
    report_path = directory / "report.json"
    write_json(report_path, report)
    started = time.monotonic()
    try:
        with isolated_store(directory, args.segment_mib << 20) as (
            address,
            reader,
            catalog,
        ):
            config_path = directory / "capture.json"
            ratio = 1.0 if args.capture else None
            if args.capture:
                write_json(
                    config_path,
                    capture_config(args, directory, address, catalog, ratio),
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
                str(
                    max(4096, args.concurrency * (args.input_len + args.output_len) * 2)
                ),
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
                "--gc-warning-threshold-secs",
                "0.02",
            ]
            if args.capture:
                server_args += ["--training-capture-config", str(config_path)]
            process = None
            with (
                (directory / "server.stdout.log").open("x") as stdout,
                (directory / "server.stderr.log").open("x") as stderr,
            ):
                try:
                    with diagnostic_server(True):
                        process = popen_launch_server(
                            args.model_path,
                            url,
                            timeout=300,
                            other_args=server_args,
                            env={
                                "SGLANG_LOG_GC": "1",
                                "SGLANG_TEST_CAPTURE_GC_LIFECYCLE": "1",
                            },
                            return_stdout_stderr=(stdout, stderr),
                        )
                    warmup(url, args)
                    control = apply_gc_policy(url, directory, args.gc_policy)
                    baseline, _ = wait_capture_idle(url)
                    verify_drained(baseline, baseline, args, directory)
                    report["observations"].append(
                        {
                            "batch": -1,
                            "elapsed_seconds": time.monotonic() - started,
                            "gc": gc_snapshot(url),
                            "processes": process_snapshot(process.pid),
                        }
                    )
                    with catalog.condition:
                        known = set(catalog.publications)
                    workload_started = time.monotonic()
                    for index in range(args.batches):
                        batch_dir = directory / f"batch-{index:03d}"
                        batch_dir.mkdir()
                        batch_args = argparse.Namespace(**vars(args))
                        batch_args.seed += index
                        batch = measure(batch_args, url, batch_dir)
                        verify_drained(
                            batch["capture_after"], baseline, args, directory
                        )
                        delta = batch["capture_counter_delta"]
                        if args.capture and (
                            delta.get("considered") != args.num_prompts
                            or delta.get("admitted", 0) == 0
                            or delta.get("admitted") != delta.get("ready")
                        ):
                            raise RuntimeError(
                                f"Incomplete capture accounting: {delta}"
                            )
                        with catalog.condition:
                            batch["publications"] = [
                                value
                                for key, value in catalog.publications.items()
                                if key not in known
                            ]
                            known = set(catalog.publications)
                            if catalog.errors:
                                raise RuntimeError(f"Catalog errors: {catalog.errors}")
                        if len(batch["publications"]) != delta.get("ready", 0):
                            raise RuntimeError(
                                "Catalog READY count differs from producer"
                            )
                        report["batches"].append(batch)
                        observation = {
                            "batch": index,
                            "elapsed_seconds": time.monotonic() - started,
                            "gc": gc_snapshot(url),
                            "processes": process_snapshot(process.pid),
                        }
                        report["observations"].append(observation)
                        write_json(report_path, report)
                        print(
                            json.dumps(
                                {
                                    "batch": index,
                                    "capture": delta,
                                    "live_request_states": observation["gc"][
                                        "live_request_states"
                                    ],
                                    "process_rss": {
                                        p["pid"]: p["rss_bytes"]
                                        for p in observation["processes"]
                                    },
                                }
                            ),
                            flush=True,
                        )
                    report["gc_lifecycle"] = verify_gc_lifecycle(
                        url, control, args.num_prompts * args.batches, directory
                    )
                    report["repeated_load_elapsed_seconds"] = (
                        time.monotonic() - workload_started
                    )
                    report["memory"] = memory_summary(
                        report["observations"], args.max_rss_growth_mib
                    )
                    report["gc_control"] = control
                    report["server_args"] = server_args
                    report["serving_elapsed_seconds"] = time.monotonic() - started
                    write_json(report_path, report)
                finally:
                    if process is not None:
                        stop_process(process)
            # Validate after exit so a producer-local buffer cannot satisfy reads.
            all_trace_ids = set()
            for index, batch in enumerate(report["batches"]):
                trace_ids = set()
                batch["readback"] = validate_publications(
                    reader, batch["publications"], args, trace_ids=trace_ids
                )
                if all_trace_ids & trace_ids:
                    raise RuntimeError("Duplicate trace ID across batches")
                all_trace_ids.update(trace_ids)
                details = json.loads(
                    (directory / f"batch-{index:03d}/requests.json").read_text()
                )
                if details["schema_version"] != 1 or details["status"] != "completed":
                    raise RuntimeError("Incomplete native client request records")
                batch["request_summary"] = summarize_requests(
                    details["requests"],
                    trace_ids,
                    count=args.num_prompts,
                    output_len=args.output_len,
                )
                write_json(report_path, report)
            if not all(item["rss_within_budget"] for item in report["memory"]):
                raise RuntimeError(
                    "Serving process RSS exceeded the declared growth budget"
                )
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--capture-slots", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 14, 27])
    parser.add_argument("--host-mib", type=int, default=256)
    parser.add_argument("--segment-mib", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-timeout", type=int, default=600)
    parser.add_argument("--max-rss-growth-mib", type=int, default=128)
    parser.add_argument(
        "--gc-policy", choices=("unchanged", "freeze-after-warmup"), default="unchanged"
    )
    args = parser.parse_args()
    if not Path(args.model_path).is_dir():
        parser.error("Use a local model directory")
    if (
        min(
            args.num_prompts,
            args.input_len,
            args.concurrency,
            args.capture_slots,
            args.host_mib,
            args.segment_mib,
            args.phase_timeout,
            args.max_rss_growth_mib,
        )
        < 1
    ):
        parser.error("Positive dimensions and budgets are required")
    if args.batches < 2 or args.output_len < 2:
        parser.error("At least two batches and two output tokens are required")
    if args.num_prompts * args.batches + args.concurrency + 1 > 16000:
        parser.error(
            "Request lifetime probe supports at most 16000 requests including warmup"
        )
    args.output_dir = args.output_dir.resolve()
    args.kv_d2h_batch_tokens = args.teacher_d2h_batch_tokens = 1
    args.device_mib = 0
    args.request_details = True
    args.latency_diagnostics = False
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
