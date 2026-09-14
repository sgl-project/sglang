"""Plan/run a bounded NCCL EP performance matrix in the selected environment.

Default is a CPU-only plan. Explicit phases launch only their own subprocesses;
no SSH, package installation, device resets or unrelated process cleanup.
"""

import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from .followup_server import REPO, save, source_head, stop_process_group


def build_plan(
    root,
    *,
    configurations=("serial", "tbo"),
    buckets=(8, 32, 64),
    layers=(4, 26),
    nccl_port=29619,
):
    from .full_model_benchmark import Workload

    Workload(tuple(buckets)).validate()
    if (
        len(configurations) < 2
        or configurations[0] != "serial"
        or len(set(configurations)) != len(configurations)
        or not set(configurations) <= {"serial", "sbo", "tbo", "sbo-tbo"}
        or not layers
        or len(set(layers)) != len(layers)
        or any(n < 2 for n in layers)
    ):
        raise ValueError(
            "Start with serial, then unique overlap configurations; layers >=2"
        )
    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
    ]
    entries = []

    def add(phase, order, config, directory, command, workload):
        entries.append(
            dict(
                phase=phase,
                order=order,
                configuration=config,
                directory=str(directory),
                command=command,
                workload=workload,
            )
        )

    for order, configs in (
        ("forward", configurations),
        ("reverse", tuple(reversed(configurations))),
    ):
        for config in configs:
            for depth in layers:
                directory = root / "pipeline" / f"layers{depth}" / order / config
                cmd = launch + [
                    "nccl_ep_test.overlap_benchmark",
                    "--graph-only",
                    "--layers",
                    str(depth),
                    "--intermediate",
                    "1408",
                    "--buckets",
                    *map(str, buckets),
                    "--samples",
                    "100",
                    "--rounds",
                    "4",
                    "--report-dir",
                    str(directory),
                ]
                cmd += [f"--{flag}" for flag in ("sbo", "tbo") if flag in config]
                add(
                    "pipeline",
                    order,
                    config,
                    directory,
                    cmd,
                    dict(layers=depth, buckets=list(buckets)),
                )
            directory = root / "model" / order / config
            cmd = launch + [
                "nccl_ep_test.full_model_benchmark",
                "--configuration",
                config,
                "--buckets",
                *map(str, buckets),
                "--nccl-port",
                str(nccl_port),
                "--reports",
                str(directory),
            ]
            add("model", order, config, directory, cmd, dict(buckets=list(buckets)))

    # Profile exactly the same implementations and capacities. Selecting a
    # bucket is done inside the worker; all buckets are still warmed/captured.
    for config in configurations:
        for bucket in dict.fromkeys((min(buckets), max(buckets))):
            for kind in ("pipeline", "model"):
                directory = root / "profile" / kind / f"B{bucket}" / config
                if kind == "pipeline":
                    module = "nccl_ep_test.overlap_benchmark"
                    args = [
                        "--graph-only",
                        "--layers",
                        str(max(layers)),
                        "--intermediate",
                        "1408",
                        "--buckets",
                        *map(str, buckets),
                        "--profile-bucket",
                        str(bucket),
                        "--samples",
                        "8",
                        "--rounds",
                        "2",
                        "--profile",
                        "--report-dir",
                        str(directory),
                    ]
                    args += [f"--{flag}" for flag in ("sbo", "tbo") if flag in config]
                else:
                    module = "nccl_ep_test.full_model_benchmark"
                    args = [
                        "--configuration",
                        config,
                        "--buckets",
                        *map(str, buckets),
                        "--profile-bucket",
                        str(bucket),
                        "--profile",
                        "--nccl-port",
                        str(nccl_port),
                        "--reports",
                        str(directory),
                    ]
                cmd = [
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx",
                    "--cuda-graph-trace=node",
                    "--sample=none",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--output",
                    str(directory / "trace"),
                    *launch,
                    module,
                    *args,
                ]
                add(
                    "profile",
                    "profile",
                    config,
                    directory,
                    cmd,
                    dict(kind=kind, bucket=bucket),
                )
    return entries


def execute(entry, *, timeout=1800):
    directory = Path(entry["directory"])
    # Partial and failed evidence stays reviewable. A retry needs a fresh root.
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "driver.log"
    record = dict(entry, source_head=source_head(), passed=False)
    env = dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS="8", NCCL_CUMEM_ENABLE="1")
    record["environment_overrides"] = {
        key: env[key] for key in ("CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_CUMEM_ENABLE")
    }
    save(directory / "invocation.json", record)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            entry["command"],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            code = process.wait(timeout=timeout)
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"Command exited {code}; inspect {log_path}")
            record["passed"] = True
        finally:
            stop_process_group(process)
            save(directory / "invocation.json", record)
            # Respect existing jobs: on OOM wait 1–5 minutes, then leave the
            # failure visible. Never clear somebody else's GPU allocations.
            if "out of memory" in log_path.read_text(errors="replace").lower():
                delay = random.randint(60, 300)
                print(f"OOM: waiting {delay}s before returning the failure", flush=True)
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    time.sleep(min(30, deadline - time.monotonic()))


def summarize(root, configurations, layers):
    from .overlap_benchmark import summarize as summarize_pipeline
    from .performance_report import compare_runs, validate_pair

    report = dict(source_head=source_head(), passed=False, pipeline=[], model=[])
    for order in ("forward", "reverse"):
        for depth in layers:
            reference_pipeline = None
            for config in configurations:
                directory = root / "pipeline" / f"layers{depth}" / order / config
                pair = [
                    json.loads((directory / f"overlap-rank{r}.json").read_text())
                    for r in (0, 1)
                ]
                summary = summarize_pipeline(pair)
                if summary["source_head"] != report["source_head"]:
                    raise ValueError("Pipeline evidence uses a different source SHA")
                normalized = dict(summary["config"])
                for field in ("sbo", "tbo", "ep_rounds_per_rank_per_forward"):
                    normalized.pop(field)
                if config == "serial":
                    reference_pipeline = (normalized, summary)
                elif normalized != reference_pipeline[0]:
                    raise ValueError("Pipeline workload differs across configurations")
                report["pipeline"].append(
                    dict(order=order, configuration=config, **summary)
                )
        reference = root / "model" / order / "serial"
        validate_pair(
            [
                json.loads((reference / f"model-rank{r}.json").read_text())
                for r in (0, 1)
            ]
        )
        for config in configurations[1:]:
            comparison = compare_runs(reference, root / "model" / order / config)
            if comparison["source_head"] != report["source_head"]:
                raise ValueError("Model evidence uses a different source SHA")
            report["model"].append(dict(order=order, **comparison))
    report["passed"] = True
    save(root / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="plan",
        choices=("plan", "pipeline", "model", "profile", "summary"),
    )
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--configurations", nargs="+", default=["serial", "tbo"])
    parser.add_argument("--buckets", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 26])
    parser.add_argument("--nccl-port", type=int, default=29619)
    args = parser.parse_args()
    root = args.reports.resolve()
    entries = build_plan(
        root,
        configurations=tuple(args.configurations),
        buckets=tuple(args.buckets),
        layers=tuple(args.layers),
        nccl_port=args.nccl_port,
    )
    if args.phase == "plan":
        print(json.dumps(dict(source_head=source_head(), entries=entries), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "summary":
        summarize(root, args.configurations, args.layers)
        return
    if args.phase == "profile" and shutil.which("nsys") is None:
        raise RuntimeError("nsys is required for the profile phase")
    if args.phase == "profile":
        # Establish unprofiled correctness/timing before paying for traces.
        summarize(root, args.configurations, args.layers)
    if args.phase in ("model", "profile"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.nccl_port))
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("Run the suite once; it starts its own torchrun workers")
    for entry in entries:
        if entry["phase"] == args.phase:
            print(f"Running {entry['directory']}", flush=True)
            execute(entry)
            if args.phase == "profile":
                from .performance_trace import export_and_analyze

                export_and_analyze(Path(entry["directory"]))


if __name__ == "__main__":
    main()
