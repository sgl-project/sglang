"""Summarize paired independent service runs written by bench_serving.py."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from summarize import require_same, software_identity, validate_parallel_layout


def server_identity(server, backend, historical=False):
    command = server["command"]
    if len(command) < 3 or command[1:3] != ["-m", "sglang.launch_server"]:
        raise ValueError("expected the recorded SGLang launch command")
    options = {}
    i = 3
    while i < len(command):
        key = command[i]
        if not key.startswith("--"):
            raise ValueError(f"unexpected launch argument: {key}")
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        elif i + 1 < len(command) and not command[i + 1].startswith("--"):
            value = command[i + 1]
            i += 2
        else:
            value = True
            i += 1
        if key in options:
            raise ValueError(f"duplicate launch argument: {key}")
        options[key] = value
    tp_size = int(options["--tp-size"])
    ep_size = int(options["--ep-size"])
    if options["--moe-a2a-backend"] != backend:
        raise ValueError("server backend disagrees with result filename")
    validate_parallel_layout(backend, tp_size, ep_size)
    # EP width and a2a backend are the independent variables. The TP baseline
    # does not use DeepEP, whose default mode in the recorded runs is auto.
    if options.pop("--deepep-mode", "auto") != "auto":
        raise ValueError("serving comparison requires DeepEP auto mode")
    del options["--ep-size"], options["--moe-a2a-backend"]
    return dict(
        interpreter=command[0],
        launch_options=options,
        metadata=server["metadata"]
        if historical
        else software_identity(server["metadata"]),
    )


def historical_servers(root):
    """Check the immutable archive; its shared snapshot is not per-run capture."""
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest:
        data = (root / entry["path"]).read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"historical evidence checksum mismatch: {entry['path']}")
    environment = json.loads((root / "revision-environment.json").read_text())
    snapshot = json.loads((root / "environment-full.json").read_text())
    return dict(revision_environment=environment, environment_snapshot=snapshot)


def summarize(directory, historical_provenance=None):
    historical = (
        historical_servers(historical_provenance)
        if historical_provenance is not None
        else None
    )
    groups = {}
    for path in sorted(directory.glob("*_r*_in*.json")):
        backend, repeat, _ = path.stem.split("_")
        run = json.loads(path.read_text())
        if backend not in ("none", "deepep"):
            raise ValueError(f"unexpected backend in {path}")
        if "server" in run:
            server = run["server"]
        elif historical is not None:
            # Only accept original client records covered by the checked archive.
            original = historical_provenance / "e2e" / path.name
            if path.read_bytes() != original.read_bytes():
                raise ValueError(f"result differs from historical archive: {path}")
            command_file = (
                historical_provenance
                / "server-commands"
                / f"{backend}_{repeat}_command.json"
            )
            server = dict(
                command=json.loads(command_file.read_text()), metadata=historical
            )
        else:
            raise ValueError(f"missing per-run server provenance: {path}")
        run["server_identity"] = server_identity(
            server, backend, historical="server" not in run
        )
        if len(run["per_request"]) != run["requests"] or any(
            r["output_tokens"] != run["output_len"] for r in run["per_request"]
        ):
            raise ValueError(f"incomplete requests: {path}")
        expected = run["requests"] * run["output_len"]
        if (
            run["output_tokens"] != expected
            or abs(run["throughput"] - expected / run["wall_seconds"]) > 1e-6
        ):
            raise ValueError(f"invalid throughput: {path}")
        runs = groups.setdefault(run["input_len"], {}).setdefault(backend, {})
        if repeat in runs:
            raise ValueError(f"duplicate repetition: {path}")
        runs[repeat] = run
    results = []
    for length, configs in sorted(groups.items()):
        if set(configs) != {"none", "deepep"}:
            raise ValueError(f"missing backend: input length {length}")
        if set(configs["none"]) != set(configs["deepep"]):
            raise ValueError(f"unpaired repetitions: input length {length}")
        for rep, tp in configs["none"].items():
            require_same(
                [tp, configs["deepep"][rep]],
                (
                    "input_sha256",
                    "seed",
                    "requests",
                    "concurrency",
                    "output_len",
                    "warmup_requests",
                    "script_sha256",
                ),
                f"input length {length}, repetition {rep}",
            )
        require_same(
            [run for runs in configs.values() for run in runs.values()],
            (
                "server_identity",
                "requests",
                "concurrency",
                "output_len",
                "warmup_requests",
                "script_sha256",
            ),
            f"input length {length}",
        )
        report = dict(input_len=length, independent_runs=len(configs["none"]))
        report["configurations"] = {}
        for backend, runs in configs.items():
            metrics = {}
            for key in (
                "throughput",
                "ttft_ms_median",
                "tpot_ms_median",
                "total_ms_median",
            ):
                values = [runs[rep][key] for rep in sorted(runs)]
                median = statistics.median(values)
                metrics[key] = dict(
                    median=median,
                    run_values=values,
                    relative_range=(max(values) - min(values)) / median,
                )
            report["configurations"][backend] = metrics
        results.append(report)
    if not results:
        raise ValueError("no serving results found")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--historical-provenance", type=Path)
    args = p.parse_args()
    args.out.write_text(
        json.dumps(summarize(args.directory, args.historical_provenance), indent=2)
        + "\n"
    )
