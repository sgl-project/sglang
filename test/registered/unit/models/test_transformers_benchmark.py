# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psutil
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts/benchmark_transformers_backend.py"
)


def test_dry_run_records_paired_fixed_length_commands_without_starting_servers(
    tmp_path,
):
    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "no-network/example",
            "--revision",
            "abc123",
            "--repeats",
            "2",
            "--dry-run",
            "--output-dir",
            str(output),
            "--server-args=--tp-size 2 --dtype bfloat16 --disable-cuda-graph",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Results:" in result.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "dry_run"
    assert len(manifest["cases"]) == 6
    assert list(output.iterdir()) == [output / "manifest.json"]
    for case in manifest["cases"]:
        server, benchmark = case["server"], case["benchmark"]
        assert server[server.index("--revision") + 1] == "abc123"
        assert server[server.index("--tp-size") + 1] == "2"
        assert benchmark[benchmark.index("--random-range-ratio") + 1] == "1"
        assert benchmark[benchmark.index("--seed") + 1] == "42"
        assert "--tokenize-prompt" in benchmark
        assert case["environment"]["SGLANG_ENABLE_TRANSFORMERS_FUSIONS"] == str(
            case["variant"] != "transformers_off"
        )
    assert [case["variant"] for case in manifest["cases"][:3]] == [
        "native",
        "transformers_off",
        "transformers_on",
    ]


def test_dry_run_rejects_server_overrides_that_break_pairing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "unused",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "output"),
            "--server-args=--model-impl auto",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
    assert not (tmp_path / "output").exists()


def test_cleanup_stops_owned_workers_across_sessions_and_preserves_unrelated_processes():
    spec = importlib.util.spec_from_file_location("transformers_benchmark", SCRIPT)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    token = uuid.uuid4().hex
    sleeper = [sys.executable, "-c", "import time; time.sleep(300)"]
    unrelated = subprocess.Popen(sleeper, start_new_session=True)
    code = "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],start_new_session=True); print(child.pid,flush=True); time.sleep(300)"
    owner = subprocess.Popen(
        [sys.executable, "-c", code],
        env={**os.environ, harness.OWNER_KEY: token},
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(owner.stdout.readline())
    try:
        harness.stop_owned(token)
        owner.wait(timeout=5)
        assert (
            not psutil.pid_exists(child_pid)
            or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
        assert unrelated.poll() is None
    finally:
        for process in (owner, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        if psutil.pid_exists(child_pid):
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
