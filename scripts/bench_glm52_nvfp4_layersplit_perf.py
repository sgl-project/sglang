"""Synthetic PD performance measurements; no dataset accuracy evaluation."""

import json
import math
import os
import statistics
import subprocess
import sys

WORKLOADS = (
    dict(name="8k_cc32", input=8192, output=512, concurrency=32, requests=256),
    dict(name="8k_cc200", input=8192, output=512, concurrency=200, requests=256),
    dict(name="32k_cc8", input=32768, output=512, concurrency=8, requests=32),
)
METRICS = ("output_throughput", "duration", "median_ttft_ms", "median_tpot_ms")


def validate_sample(sample, workload):
    if (
        sample["completed"] != workload["requests"]
        or sample["total_input_tokens"] != workload["requests"] * workload["input"]
        or sample["total_output_tokens"] != workload["requests"] * workload["output"]
        or any(sample.get("errors", []))
        or any(not math.isfinite(sample[key]) or sample[key] <= 0 for key in METRICS)
    ):
        raise RuntimeError(f"Incomplete or invalid performance run: {workload['name']}")


def summarize_performance(runs, allocations):
    result = {"accuracy_evaluated": False, "workloads": {}, "memory": {}}
    for workload in WORKLOADS:
        medians = {}
        for phase in ("baseline", "layer_split"):
            samples = runs[phase][workload["name"]]
            if len(samples) != 3:
                raise RuntimeError("Each performance workload requires three runs")
            for sample in samples:
                validate_sample(sample, workload)
            medians[phase] = {
                key: statistics.median(sample[key] for sample in samples)
                for key in METRICS
            }
        result["workloads"][workload["name"]] = {
            "configuration": workload,
            "medians": medians,
            "layer_split_change_percent": {
                key: 100 * (medians["layer_split"][key] / medians["baseline"][key] - 1)
                for key in METRICS
            },
        }
    pools = {}
    byte_fields = (
        "main_bytes",
        "index_bytes",
        "remote_main_bytes",
        "remote_index_bytes",
        "global_scale_bytes",
    )
    for phase in ("baseline", "layer_split"):
        records = allocations[phase]
        layers = max(record["layers"] for record in records)
        pools[phase] = {
            record["device"]: record for record in records if record["layers"] == layers
        }
        if len(pools[phase]) != 4 or layers <= 1:
            raise RuntimeError("Expected four Prefill target pools")
        for record in pools[phase].values():
            if record["total_bytes"] != sum(record[key] for key in byte_fields):
                raise RuntimeError("Pool allocation byte accounting does not match")
            if record["layer_split"] != (phase == "layer_split"):
                raise RuntimeError("Pool allocation has the wrong layer-split setting")
    if pools["baseline"].keys() != pools["layer_split"].keys():
        raise RuntimeError("Prefill device sets do not match")
    for device, before in pools["baseline"].items():
        after = pools["layer_split"][device]
        if (
            before["capacity_tokens"] != after["capacity_tokens"]
            or before["layers"] != after["layers"]
            or after["total_bytes"] >= before["total_bytes"]
        ):
            raise RuntimeError("Pool capacity differs or memory did not decrease")
        result["memory"][device] = {
            "baseline": before,
            "layer_split": after,
            "reduction_percent": 100
            * (1 - after["total_bytes"] / before["total_bytes"]),
        }
    return result


def run_synthetic_phase(args, phase, router_url, worker_urls, control, report):
    import requests
    from bench_glm52_nvfp4_layersplit_slurm import check_peer, write_json

    samples = report.setdefault("synthetic_runs", {}).setdefault(phase, {})
    for workload in WORKLOADS:
        samples[workload["name"]] = []
        for round_index in range(3):
            check_peer(control)
            for url in worker_urls:
                requests.post(
                    url + "/flush_cache", params={"timeout": 60}, timeout=65
                ).raise_for_status()
            name = f"{phase}.{workload['name']}.run{round_index + 1}"
            output = args.output_dir / f"{name}.jsonl"
            command = [
                sys.executable,
                "-m",
                "sglang.benchmark.serving",
                "--backend",
                "sglang",
                "--base-url",
                router_url,
                "--model",
                args.model_path,
                "--tokenizer",
                args.model_path,
                "--dataset-name",
                "random",
                "--random-input-len",
                str(workload["input"]),
                "--random-output-len",
                str(workload["output"]),
                "--random-range-ratio",
                "1",
                "--max-concurrency",
                str(workload["concurrency"]),
                "--num-prompts",
                str(workload["requests"]),
                "--seed",
                "42",
                "--tokenize-prompt",
                "--warmup-requests",
                "0",
                "--ready-check-timeout-sec",
                "0",
                "--disable-tqdm",
                "--output-details",
                "--output-file",
                str(output),
            ]
            if dataset_path := os.environ.get("GLM52_DATASET_PATH"):
                command += ["--dataset-path", dataset_path]
            write_json(args.output_dir / f"{name}.command.json", command)
            with (args.output_dir / f"{name}.log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            check_peer(control)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if len(rows) != 1:
                raise RuntimeError(f"Expected one benchmark result: {output}")
            validate_sample(rows[0], workload)
            samples[workload["name"]].append(rows[0])
            write_json(args.output_dir / "results.json", report)
