#!/usr/bin/env python3
"""Run the six progressive PD-flip trials with marker-based coordination."""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


MODES = ("full", "partial", "zero")
PATHS = ("recovery", "commit")


def run_shell(command, *, env, log_path=None):
    output = None
    if log_path:
        output = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.run(
            command,
            shell=True,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT if output else None,
            check=False,
        ).returncode
    finally:
        if output:
            output.close()


def wait_for_path(path, timeout, interval):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(interval)
    raise TimeoutError("timed out waiting for %s" % path)


def wait_for_store(url, timeout, interval):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=min(timeout, 3.0)) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(interval)
    raise TimeoutError("dedicated store did not become ready: %r" % last_error)


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_case(args, mode, decision_path):
    case_dir = Path(args.output_root) / decision_path / mode
    case_dir.mkdir(parents=True, exist_ok=True)
    ready = case_dir / "workload.ready.json"
    done = case_dir / "controller.done"
    for marker in (ready, done):
        marker.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MODE": mode,
            "DECISION_PATH": decision_path,
            "OUTPUT_DIR": str(case_dir.resolve()),
            "READY_MARKER": str(ready.resolve()),
            "CONTROLLER_DONE_MARKER": str(done.resolve()),
        }
    )
    if run_shell(args.reset_store_cmd, env=env) != 0:
        raise RuntimeError("store reset command failed")
    wait_for_store(args.store_ready_url, args.timeout_seconds, args.poll_interval_seconds)

    measure_log = (case_dir / "measurement.log").open("w", encoding="utf-8")
    measurement = subprocess.Popen(
        args.measure_command,
        shell=True,
        env=env,
        stdout=measure_log,
        stderr=subprocess.STDOUT,
    )
    workload_command = [
        sys.executable,
        str(Path(__file__).with_name("pd_flip_progressive_workload.py")),
        "--base-url",
        args.base_url,
        "--source-url",
        args.source_url,
        "--target-url",
        args.target_url,
        "--admin-api-key-env",
        args.admin_api_key_env,
        "--ready-marker",
        str(ready),
        "--controller-done-marker",
        str(done),
        "--model",
        args.model,
        "--mode",
        mode,
        "--decision-path",
        decision_path,
        "--output-dir",
        str(case_dir),
        "--coordination-timeout-seconds",
        str(args.timeout_seconds),
    ]
    workload_log = (case_dir / "workload.log").open("w", encoding="utf-8")
    workload = subprocess.Popen(
        workload_command,
        env=env,
        stdout=workload_log,
        stderr=subprocess.STDOUT,
    )
    controller_rc = workload_rc = 1
    try:
        wait_for_path(ready, args.timeout_seconds, args.poll_interval_seconds)
        controller_rc = run_shell(
            args.controller_command,
            env=env,
            log_path=case_dir / "controller.log",
        )
        done.touch()
        workload_rc = workload.wait(timeout=args.timeout_seconds)
    finally:
        done.touch()
        terminate(workload)
        terminate(measurement)
        workload_log.close()
        measure_log.close()
    if controller_rc or workload_rc:
        raise RuntimeError(
            "case %s/%s failed: controller=%s workload=%s"
            % (decision_path, mode, controller_rc, workload_rc)
        )
    if run_shell(
        args.summarize_command,
        env=env,
        log_path=case_dir / "summarize.log",
    ):
        raise RuntimeError("summarizer failed for %s/%s" % (decision_path, mode))


def run(args):
    if not args.reset_store_cmd.strip():
        raise ValueError("--reset-store-cmd must be non-empty")
    if not os.environ.get(args.admin_api_key_env):
        raise ValueError("admin key environment variable is empty")
    for decision_path in PATHS:
        for mode in MODES:
            run_case(args, mode, decision_path)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--admin-api-key-env", default="ADMIN_API_KEY")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reset-store-cmd", required=True)
    parser.add_argument("--store-ready-url", required=True)
    parser.add_argument("--measure-command", required=True)
    parser.add_argument("--controller-command", required=True)
    parser.add_argument("--summarize-command", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
