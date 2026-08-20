#!/usr/bin/env python3
"""Run three complete, alternating MiniMax-M3 MSA A/B repetitions."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from summarize_msa_repetitions import expected_order

OFFLINE_THROUGHPUT_DATASET = "random-ids"
OFFLINE_THROUGHPUT_ARGS = (
    "--dataset-name",
    OFFLINE_THROUGHPUT_DATASET,
    "--tokenize-prompt",
)


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def server_healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/health_generate", timeout=timeout
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "<server log does not exist>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def wait_for_server(
    process: subprocess.Popen, base_url: str, log_path: Path, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited with status {return_code} before readiness:\n"
                + tail(log_path)
            )
        if server_healthy(base_url, timeout=5):
            return
        time.sleep(5)
    raise TimeoutError(f"server was not ready after {timeout}s:\n" + tail(log_path))


def validate_provider_log(log_path: Path, provider: str) -> None:
    wanted = (
        f"main_attn={provider}",
        "msa_decode=True",
        "msa_owns_decode=True",
        "decode_cuda_graph=True",
    )
    lines = log_path.read_text(errors="replace").splitlines()
    if not any(all(fragment in line for fragment in wanted) for line in lines):
        raise RuntimeError(
            "server did not confirm the requested MSA provider and decode graph state; "
            f"wanted {wanted}:\n{tail(log_path)}"
        )


def stop_server(process: subprocess.Popen, base_url: str) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not server_healthy(base_url):
            return
        time.sleep(2)
    raise RuntimeError(f"server still answers at {base_url} after process shutdown")


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"+ {command_text(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def server_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--tp",
        "4",
        "--attention-backend",
        "fa4",
        "--page-size",
        "128",
        "--moe-runner-backend",
        "deep_gemm",
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.75",
        "--disable-prefill-cuda-graph",
        "--host",
        args.host,
        "--port",
        str(args.port),
        *args.extra_server_arg,
    ]


def run_provider(
    *,
    args: argparse.Namespace,
    repo: Path,
    environment: dict[str, str],
    repetition_dir: Path,
    provider_role: str,
) -> None:
    provider = "fmha_sm100" if provider_role == "baseline" else "flashinfer"
    label = "external" if provider_role == "baseline" else "flashinfer"
    output_dir = repetition_dir / provider_role
    server_log = repetition_dir / f"{provider_role}_server.log"
    warmup_output = repetition_dir / f"{provider_role}_warmup.jsonl"
    if server_healthy(args.base_url):
        raise RuntimeError(f"another server is already answering at {args.base_url}")

    server_env = dict(environment)
    server_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH": "1",
            "SGLANG_MINIMAX_MSA_BACKEND": provider,
        }
    )
    launch_command = server_command(args)
    print(f"+ {provider}: {command_text(launch_command)}", flush=True)
    with server_log.open("w") as log:
        process = subprocess.Popen(
            launch_command,
            cwd=repo,
            env=server_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(process, args.base_url, server_log, args.server_timeout)
            log.flush()
            validate_provider_log(server_log, provider)
            warmup_command = [
                args.python,
                "-m",
                "sglang.benchmark.serving",
                "--backend",
                "sglang",
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                *OFFLINE_THROUGHPUT_ARGS,
                "--num-prompts",
                "1",
                "--random-input-len",
                "8192",
                "--random-output-len",
                "1024",
                "--random-range-ratio",
                "1",
                "--request-rate",
                "inf",
                "--max-concurrency",
                "1",
                "--seed",
                "20260819",
                "--flush-cache",
                "--output-file",
                str(warmup_output),
            ]
            run_checked(warmup_command, cwd=repo, env=environment)

            gate_env = dict(environment)
            gate_env.update(
                {
                    "BASE_URL": args.base_url,
                    "MODEL": args.model,
                    "LABEL": label,
                    "OUTPUT_DIR": str(output_dir),
                    "LONGBENCH_SUBSET": str(args.longbench_subset.resolve()),
                    "GPQA_DATASET": str(args.gpqa_dataset.resolve()),
                    "NUM_THREADS": str(args.num_threads),
                    "SERVER_LOG": str(server_log),
                    "SERVING_DATASET_NAME": OFFLINE_THROUGHPUT_DATASET,
                }
            )
            run_checked(
                ["bash", "benchmark/minimax_m3/run_msa_gate.sh"],
                cwd=repo,
                env=gate_env,
            )
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited during {provider_role} gate:\n" + tail(server_log)
                )
        finally:
            stop_server(process, args.base_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--longbench-subset", type=Path, required=True)
    parser.add_argument("--gpqa-dataset", type=Path, required=True)
    parser.add_argument("--flashinfer-source-dir", type=Path, required=True)
    parser.add_argument("--expected-flashinfer-head", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--expected-tvm-ffi-version", required=True)
    parser.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        help="Append one launch_server argument; repeat for multiple arguments",
    )
    parser.add_argument("--min-median-output-throughput-gain", type=float, default=0.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"

    repo = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    environment = dict(os.environ)
    python_path = str((repo / "python").resolve())
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment.update(
        {
            "PYTHONPATH": python_path,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    preflight_command = [
        args.python,
        "benchmark/minimax_m3/probe_msa_e2e_dependencies.py",
        "--model",
        args.model,
        "--longbench-subset",
        str(args.longbench_subset.resolve()),
        "--gpqa-dataset",
        str(args.gpqa_dataset.resolve()),
        "--flashinfer-source-dir",
        str(args.flashinfer_source_dir.resolve()),
        "--expected-flashinfer-head",
        args.expected_flashinfer_head,
        "--expected-tvm-ffi-version",
        args.expected_tvm_ffi_version,
        "--output",
        str(output_root / "preflight.json"),
    ]
    run_checked(preflight_command, cwd=repo, env=environment)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "base_url": args.base_url,
        "flashinfer_source_dir": str(args.flashinfer_source_dir.resolve()),
        "expected_flashinfer_head": args.expected_flashinfer_head,
        "expected_tvm_ffi_version": args.expected_tvm_ffi_version,
        "server_command": server_command(args),
        "repetitions": 3,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    for repetition in range(1, 4):
        repetition_dir = output_root / f"rep{repetition:02d}"
        repetition_dir.mkdir()
        order = expected_order(repetition)
        (repetition_dir / "order.json").write_text(
            json.dumps({"order": order}, indent=2) + "\n"
        )
        for provider_role in order:
            run_provider(
                args=args,
                repo=repo,
                environment=environment,
                repetition_dir=repetition_dir,
                provider_role=provider_role,
            )
        run_checked(
            [
                args.python,
                "benchmark/minimax_m3/compare_msa_gate.py",
                "--baseline-dir",
                str(repetition_dir / "baseline"),
                "--candidate-dir",
                str(repetition_dir / "candidate"),
                "--output",
                str(repetition_dir / "comparison.json"),
            ],
            cwd=repo,
            env=environment,
        )

    run_checked(
        [
            args.python,
            "benchmark/minimax_m3/summarize_msa_repetitions.py",
            "--root",
            str(output_root),
            "--min-median-output-throughput-gain",
            str(args.min_median_output_throughput_gain),
        ],
        cwd=repo,
        env=environment,
    )


if __name__ == "__main__":
    main()
