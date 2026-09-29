#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

"""Compare native and Transformers generation with fresh servers on a local GPU."""

import argparse
import importlib.metadata
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OWNER_KEY = "SGLANG_TRANSFORMERS_BENCHMARK_OWNER"
VARIANTS = (
    ("native", "sglang", True),
    ("transformers_off", "transformers", False),
    ("transformers_on", "transformers", True),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Checkpoint directory or Hugging Face model ID"
    )
    parser.add_argument(
        "--revision",
        help="Checkpoint revision; remote revisions resolve to a commit before execution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results")
        / time.strftime("transformers-%Y%m%dT%H%M%SZ", time.gmtime()),
    )
    for name, default in (
        ("repeats", 3),
        ("num-prompts", 64),
        ("input-len", 512),
        ("output-len", 128),
        ("concurrency", 8),
        ("seed", 42),
        ("port", 30000),
        ("startup-timeout", 900),
        ("benchmark-timeout", 1800),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument(
        "--server-args",
        default="",
        help="Quoted extra launch_server arguments: TP, dtype, attention backend, memory/context/token limits, quantization, CUDA graphs, compile, radix cache, or remote code",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands and manifest without loading a model or starting servers",
    )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if type(value) is int and name not in {"seed", "port"} and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit an unsigned 32-bit integer")
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag, kind in {
        "tp-size": int,
        "dtype": str,
        "attention-backend": str,
        "prefill-attention-backend": str,
        "decode-attention-backend": str,
        "cuda-graph-backend-prefill": str,
        "cuda-graph-backend-decode": str,
        "cuda-graph-tc-compiler": str,
        "mem-fraction-static": float,
        "context-length": int,
        "chunked-prefill-size": int,
        "max-total-tokens": int,
        "quantization": str,
    }.items():
        extra.add_argument(f"--{flag}", type=kind)
    for flag in (
        "disable-cuda-graph",
        "enable-torch-compile",
        "disable-radix-cache",
        "trust-remote-code",
    ):
        extra.add_argument(f"--{flag}", action="store_true")
    args.extra = shlex.split(args.server_args)
    extra.parse_args(args.extra)
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def command(module, **options):
    result = [sys.executable, "-m", module]
    for key, value in options.items():
        if value is not None:
            result += ["--" + key.replace("_", "-")]
            if value is not True:
                result.append(str(value))
    return result


def commands(args, output, implementation, revision):
    server = (
        command(
            "sglang.launch_server",
            model_path=args.model,
            host="127.0.0.1",
            port=args.port,
            model_impl=implementation,
            random_seed=args.seed,
            revision=revision,
        )
        + args.extra
    )
    benchmark = command(
        "sglang.benchmark.serving",
        backend="sglang",
        base_url=f"http://127.0.0.1:{args.port}",
        model=args.model,
        tokenizer=args.output_dir / "tokenizer",
        dataset_name="random-ids",
        tokenize_prompt=True,
        random_input_len=args.input_len,
        random_output_len=args.output_len,
        random_range_ratio=1,
        num_prompts=args.num_prompts,
        max_concurrency=args.concurrency,
        request_rate="inf",
        seed=args.seed,
        temperature=0,
        warmup_requests=args.concurrency,
        flush_cache=True,
        disable_tqdm=True,
        output_details=True,
        output_file=output / "metrics.jsonl",
    )
    return server, benchmark


def stop_owned(token):
    import psutil

    def owned():
        found = []
        for process in psutil.process_iter():
            try:
                if process.environ().get(OWNER_KEY) == token:
                    found.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return found

    for operation, timeout in (("terminate", 10), ("kill", 5)):
        processes = owned()
        for process in processes:
            try:
                getattr(process, operation)()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(processes, timeout=timeout)
    if owned():
        raise RuntimeError("Benchmark-owned worker processes survived cleanup")


def wait_ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}")
        try:
            with urlopen(url + "/health_generate", timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("Server readiness timed out")


def execute(args, case, environment):
    output = Path(case["directory"])
    output.mkdir()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    token = uuid.uuid4().hex
    environment = {**environment, **case["environment"], OWNER_KEY: token}
    try:
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(
                case["server"],
                env=environment,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(server, case["url"], args.startup_timeout)
            if (
                case["variant"] == "native"
                and "Using Transformers backend." in (output / "server.log").read_text()
            ):
                raise RuntimeError(
                    "The native baseline fell back to Transformers; choose a native-supported checkpoint"
                )
            for endpoint in ("server_info", "model_info"):
                with urlopen(case["url"] + "/" + endpoint, timeout=30) as response:
                    write_json(output / f"{endpoint}.json", json.load(response))
            with (output / "benchmark.log").open("w") as benchmark_log:
                client = subprocess.Popen(
                    case["benchmark"],
                    env=environment,
                    cwd=ROOT,
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = client.wait(timeout=args.benchmark_timeout)
                if code:
                    raise subprocess.CalledProcessError(code, case["benchmark"])
            metrics = json.loads(
                (output / "metrics.jsonl").read_text().splitlines()[-1]
            )
            if metrics.get("completed") != args.num_prompts or any(
                metrics.get("errors", [])
            ):
                raise RuntimeError(
                    "Benchmark did not complete every request successfully"
                )
    finally:
        stop_owned(token)


def main():
    args = parse_args()
    if Path(args.model).is_dir():
        args.model = str(Path(args.model).resolve())
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    versions = {}
    for (
        package
    ) in "sglang torch transformers triton sgl-kernel flashinfer-python".split():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    def git(*options):
        return subprocess.check_output(["git", *options], cwd=ROOT, text=True).strip()

    manifest = {
        "arguments": vars(args),
        "python": sys.version,
        "versions": versions,
        "git_sha": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "status": "planned",
        "cases": [],
    }
    path = args.output_dir / "manifest.json"
    write_json(path, manifest)
    try:
        revision = args.revision
        if not args.dry_run:
            from transformers import AutoTokenizer

            try:
                manifest["gpu"] = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,uuid,driver_version,memory.total",
                        "--format=csv",
                    ],
                    text=True,
                    stderr=subprocess.STDOUT,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                manifest["gpu"] = repr(error)
            if not Path(args.model).is_dir():
                from huggingface_hub import HfApi

                revision = HfApi().model_info(args.model, revision=revision).sha
            AutoTokenizer.from_pretrained(
                args.model,
                revision=revision,
                trust_remote_code="--trust-remote-code" in args.extra,
            ).save_pretrained(args.output_dir / "tokenizer")
        manifest["resolved_revision"] = revision
        for repeat in range(args.repeats):
            for variant, implementation, fusions in (
                VARIANTS[repeat % 3 :] + VARIANTS[: repeat % 3]
            ):
                output = args.output_dir / f"{repeat + 1:02d}-{variant}"
                server, benchmark = commands(args, output, implementation, revision)
                manifest["cases"].append(
                    {
                        "variant": variant,
                        "repeat": repeat + 1,
                        "directory": str(output),
                        "url": f"http://127.0.0.1:{args.port}",
                        "server": server,
                        "benchmark": benchmark,
                        "environment": {
                            "SGLANG_ENABLE_TRANSFORMERS_FUSIONS": str(fusions),
                            "SGLANG_TRANSFORMERS_DISABLED_FUSIONS": "",
                        },
                        "status": "planned",
                    }
                )
        write_json(path, manifest)
        for case in manifest["cases"]:
            print(case["variant"], shlex.join(case["server"]), flush=True)
            print(shlex.join(case["benchmark"]), flush=True)
            if not args.dry_run:
                execute(args, case, environment)
                case["status"] = "completed"
                write_json(path, manifest)
        manifest["status"] = "dry_run" if args.dry_run else "completed"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(path, manifest)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    main()
