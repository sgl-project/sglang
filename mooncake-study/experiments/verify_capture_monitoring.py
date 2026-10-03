"""Real capture traffic and control transitions for a monitoring-stack check."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
import torch
from benchmark_training_capture import (
    capture_config,
    free_port,
    isolated_store,
    stop_process,
    validate_publications,
    wait_capture_idle,
    write_json,
)
from prometheus_client.parser import text_string_to_metric_families
from sglang.srt.training_capture.metrics import CaptureMetrics
from sglang.test.test_utils import popen_launch_server


def verify_metrics(url, state, output):
    expected = {
        ("device_allocated_bytes", None): state["host_pool"]["device_allocated_bytes"],
        ("device_limit_bytes", None): state["host_pool"]["device_limit_bytes"],
        ("kv_staging_allocated_bytes", None): state["host_pool"][
            "device_allocated_bytes"
        ],
        ("kv_staging_limit_bytes", None): state["host_pool"]["device_limit_bytes"],
        ("admission_paused", None): int(state["admission_paused"]),
        ("disabled", None): 0,
    }
    for destination in ("host", "device"):
        expected["kv_export_enqueued_bytes_total", destination] = state["host_pool"][
            f"kv_export_{destination}_enqueued_bytes"
        ]
    if "request_router" in state:
        for event in CaptureMetrics.ROUTING_EVENTS:
            expected["routing_events_total", event] = state["request_router"].get(
                event, 0
            )
    deadline = time.monotonic() + 15
    while True:
        response = requests.get(url + "/metrics", timeout=10)
        response.raise_for_status()
        actual = {}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if any(
                    sample.labels.get(rank, "0") != "0"
                    for rank in ("tp_rank", "pp_rank")
                ):
                    continue
                key = (
                    sample.name.removeprefix("sglang:training_capture_"),
                    sample.labels.get("destination") or sample.labels.get("event"),
                )
                if key in expected:
                    if key in actual:
                        raise RuntimeError("Expected one producer per metric series")
                    actual[key] = sample.value
        if actual == expected:
            output.write_text(response.text)
            return {
                name + (":" + destination if destination else ""): value
                for (name, destination), value in actual.items()
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Prometheus export disagrees with status: {actual} != {expected}"
            )
        time.sleep(0.2)


def wait_available(url, count):
    deadline = time.monotonic() + 60
    while True:
        state, _ = wait_capture_idle(url)
        if (
            state["states"].get("available", 0) >= count
            and state["admission"]["effective_ratio"] == 1
        ):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Capture admission did not recover: {state}")
        time.sleep(0.1)


def run(args):
    root = args.output_dir
    url = f"http://127.0.0.1:{args.port}"
    report = {
        "status": "running",
        "config": vars(args) | {"output_dir": str(root)},
        "scope": f"Real TP{args.tp_size} SGLang and local TCP Mooncake; test Catalog, synthetic fixed-length traffic. Monitoring correctness, not SLO acceptance.",
        "observations": [],
        "requests": 0,
        "source_sha256": {},
    }
    checkout = Path(__file__).resolve().parents[2]
    for name in (
        "python/sglang/srt/training_capture/metrics.py",
        "examples/monitoring/grafana/dashboards/json/training-capture-dashboard.json",
        "mooncake-study/experiments/verify_capture_monitoring.py",
    ):
        report["source_sha256"][name] = hashlib.sha256(
            (checkout / name).read_bytes()
        ).hexdigest()
    process = None
    try:
        with isolated_store(root, args.segment_mib << 20) as (address, reader, catalog):
            config = capture_config(args, root, address, catalog, 1.0)
            config["adaptive"] = {
                "latency": {
                    "ttft_seconds": 60.0,
                    "tpot_seconds": 60.0,
                    "min_observations": 1,
                    "window_seconds": 120.0,
                }
            }
            write_json(root / "capture.json", config)
            process = popen_launch_server(
                args.model_path,
                url,
                timeout=300,
                other_args=[
                    "--tp-size",
                    str(args.tp_size),
                    "--training-capture-config",
                    str(root / "capture.json"),
                    "--enable-metrics",
                    "--served-model-name",
                    "capture-monitoring-qwen3",
                    "--admin-api-key",
                    "capture-monitoring-test-admin",
                    "--skip-server-warmup",
                    "--skip-tokenizer-init",
                    "--attention-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.25",
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "8",
                    "--chunked-prefill-size",
                    "128",
                    "--cuda-graph-max-bs-decode",
                    "8",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                ],
            )
            wait_available(url, args.concurrency)
            start = time.monotonic()
            write_json(
                root / "ready.json",
                {
                    "url": url,
                    "pid": process.pid,
                    "start_unix": time.time(),
                    "hold_seconds": args.hold_seconds,
                    "source_sha256": report["source_sha256"],
                },
            )
            phase, step = None, 0
            captured_ids = set()
            while (elapsed := time.monotonic() - start) < args.hold_seconds:
                current = (
                    "capture"
                    if elapsed < args.hold_seconds / 3
                    else "paused"
                    if elapsed < 2 * args.hold_seconds / 3
                    else "resumed"
                )
                if current != phase:
                    if current != "capture":
                        response = requests.post(
                            url + "/control_training_capture",
                            json={
                                "action": "pause" if current == "paused" else "resume"
                            },
                            headers={
                                "Authorization": "Bearer capture-monitoring-test-admin"
                            },
                            timeout=15,
                        )
                        response.raise_for_status()
                        if not response.json()["success"]:
                            raise RuntimeError(response.text)
                        if current == "resumed":
                            wait_available(url, args.concurrency)
                    phase = current
                ids = [f"monitor-{step}-{row}" for row in range(args.concurrency)]
                streaming = step % 2 == 0
                response = requests.post(
                    url + "/generate",
                    json={
                        "rid": ids,
                        "stream": streaming,
                        "input_ids": [
                            [1000 + step * args.concurrency + row]
                            + [200] * (args.input_len - 1)
                            for row in range(args.concurrency)
                        ],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": args.output_len,
                            "ignore_eos": True,
                        },
                    },
                    stream=streaming,
                    timeout=60,
                )
                response.raise_for_status()
                if streaming:
                    latest, done = {}, False
                    for line in response.iter_lines():
                        if not line.startswith(b"data: "):
                            continue
                        data = line.removeprefix(b"data: ")
                        if data == b"[DONE]":
                            done = True
                            break
                        item = json.loads(data)
                        latest[item["index"]] = item
                    response.close()
                    if not done or set(latest) != set(range(len(ids))):
                        raise RuntimeError("Incomplete monitoring response stream")
                    outputs = list(latest.values())
                else:
                    outputs = response.json()
                if len(outputs) != len(ids) or any(
                    row["meta_info"]["completion_tokens"] != args.output_len
                    for row in outputs
                ):
                    raise RuntimeError("Monitoring requests did not finish")
                if phase != "paused":
                    captured_ids.update(
                        hashlib.sha256(rid.encode()).hexdigest() for rid in ids
                    )
                report["requests"] += len(ids)
                state, _ = wait_capture_idle(url)
                if state["disabled_reason"] is not None or state[
                    "admission_paused"
                ] != (phase == "paused"):
                    raise RuntimeError(f"Unexpected capture state: {state}")
                if state["counters"].get("ready", 0) != len(captured_ids):
                    raise RuntimeError("Capture count disagrees with control phase")
                metrics = verify_metrics(url, state, root / f"metrics-{step:03d}.prom")
                observation = {
                    "unix": time.time(),
                    "phase": phase,
                    "streaming": streaming,
                    "state": state,
                    "metrics": metrics,
                }
                report["observations"].append(observation)
                with (root / "timeline.jsonl").open("a") as stream:
                    stream.write(json.dumps(observation) + "\n")
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "step": step,
                            "ready": state["counters"].get("ready", 0),
                        }
                    ),
                    flush=True,
                )
                step += 1
                time.sleep(args.interval_seconds)
            final, _ = wait_capture_idle(url)
            publications = list(catalog.publications.values())
            if catalog.errors:
                raise RuntimeError(f"Catalog errors: {catalog.errors}")
            stop_process(process)
            process = None
            trace_ids = set()
            report["readback"] = validate_publications(
                reader, publications, args, trace_ids=trace_ids
            )
            if trace_ids != captured_ids or len(trace_ids) != final["counters"].get(
                "ready", 0
            ):
                raise RuntimeError(
                    "Published snapshots do not match generated requests"
                )
            if {row["phase"] for row in report["observations"]} != {
                "capture",
                "paused",
                "resumed",
            }:
                raise RuntimeError("Monitoring phases incomplete")
            if any(
                value
                for name, value in final["counters"].items()
                if "failed" in name or "error" in name
            ):
                raise RuntimeError(f"Capture failed: {final['counters']}")
            if any(
                final["host_pool"][f"kv_export_{target}_enqueued_bytes"] <= 0
                for target in ("host", "device")
            ):
                raise RuntimeError("Both HiCache export destinations must be exercised")
            report.update(status="completed", final=final)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None:
            stop_process(process)
        write_json(root / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--hold-seconds", type=float, default=180)
    parser.add_argument("--interval-seconds", type=float, default=4)
    args = parser.parse_args()
    if (
        args.hold_seconds < 30
        or args.interval_seconds <= 0
        or not 0 <= args.port < 65536
    ):
        parser.error(
            "Hold at least 30 seconds with a positive interval and a valid port"
        )
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.port = args.port or free_port()
    args.input_len, args.output_len, args.concurrency = 16, 32, 4
    args.layers, args.seed = [0, 14, 27], 42
    args.capture_slots, args.host_mib, args.device_mib, args.segment_mib = (
        16,
        64,
        16,
        1024,
    )
    args.kv_export_backend, args.kv_d2h_batch_tokens, args.teacher_d2h_batch_tokens = (
        "hicache",
        8,
        8,
    )
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
