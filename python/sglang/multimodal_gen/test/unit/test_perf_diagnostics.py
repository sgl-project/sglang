import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil

from sglang.multimodal_gen.test.runner import perf_diagnostics, pytest_runner


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv(perf_diagnostics._ROOT_ENV, raising=False)
    monkeypatch.delenv(perf_diagnostics._ATTEMPT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    diagnostics = perf_diagnostics.AttemptDiagnostics(1)
    diagnostics.start(os.getpid())
    diagnostics.observe(b"BEGIN diffusion testcase: test\n")
    perf_diagnostics.record_request({"e2e_ms": 10})
    diagnostics.finish(1)
    assert diagnostics.directory is None
    assert list(tmp_path.iterdir()) == []


def test_process_sampling_continues_while_nvml_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv(perf_diagnostics._ROOT_ENV, str(tmp_path))
    entered = threading.Event()
    release = threading.Event()

    def blocked_init():
        entered.set()
        assert release.wait(10)
        raise perf_diagnostics.pynvml.NVMLError_NotSupported()

    monkeypatch.setattr(perf_diagnostics.pynvml, "nvmlInit", blocked_init)
    diagnostics = perf_diagnostics.AttemptDiagnostics(1)
    try:
        diagnostics.start(os.getpid())
        assert entered.wait(5)
        path = diagnostics.directory / "processes.jsonl"
        deadline = time.monotonic() + 5
        samples = []
        while time.monotonic() < deadline:
            if path.exists():
                # another thread may currently be writing the last record
                lines = path.read_text().splitlines(keepends=True)
                samples = [json.loads(line) for line in lines if line.endswith("\n")]
            if len(samples) >= 2:
                break
            time.sleep(0.05)
        assert len(samples) >= 2
        assert not release.is_set()
        assert all(
            any(p["pid"] == os.getpid() and "kernel" in p for p in s["processes"])
            for s in samples
        )
    finally:
        release.set()
        diagnostics.finish(0)
    assert not diagnostics.thread.is_alive()
    assert not diagnostics.process_thread.is_alive()
    assert _events(diagnostics.directory / "events.jsonl")[-1][
        "process_sampler_stopped"
    ]


def test_fragmented_boundaries_do_not_save_arbitrary_logs(monkeypatch, tmp_path):
    monkeypatch.setenv(perf_diagnostics._ROOT_ENV, str(tmp_path))
    diagnostics = perf_diagnostics.AttemptDiagnostics(3)
    diagnostics.observe(b"arbitrary secret: do-not-save\nBEGIN diffusion test")
    diagnostics.observe(b"case: case-1\nAll workers are ready\n[DecodingStage] sta")
    diagnostics.observe(b"rted\n[DecodingStage] finished in 1 seconds\n")
    diagnostics.finish(1)
    events = _events(diagnostics.directory / "events.jsonl")
    assert events[0]["attempt"] == 3
    assert [e["boundary"] for e in events if e["event"] == "observed_boundary"] == [
        "case_begin",
        "workers_ready",
        "stage_begin",
        "stage_end",
    ]
    assert events[-1]["returncode"] == 1
    assert "do-not-save" not in (diagnostics.directory / "events.jsonl").read_text()


def test_real_process_counters():
    sample = perf_diagnostics._process_sample(psutil.Process())
    assert sample["pid"] == os.getpid()
    assert sample["rss_bytes"] > 0
    assert sample["minor_faults"] >= 0
    assert sample["major_faults"] >= 0
    assert sample["io"]["read_bytes"] >= 0
    assert set(sample["kernel"]) == {"syscall", "wchan"}
    assert sample["scheduler"]["cpu"] >= 0
    assert sample["scheduler"]["schedstat"]["runtime_ns"] > 0
    assert sample["scheduler"]["schedstat"]["runqueue_wait_ns"] >= 0


def test_scheduler_sample(monkeypatch):
    values = {
        "/proc/123/schedstat": "1200 3400 56\n",
        "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq": "3200000\n",
    }
    monkeypatch.setattr(
        perf_diagnostics.Path, "read_text", lambda path: values[str(path)]
    )
    assert perf_diagnostics._scheduler_sample(123, 7) == {
        "cpu": 7,
        "schedstat": {"runtime_ns": 1200, "runqueue_wait_ns": 3400, "timeslices": 56},
        "frequency_khz": 3200000,
    }


def test_optional_scheduler_files_preserve_process_counters(monkeypatch):
    original = perf_diagnostics.Path.read_text

    def read(path):
        if path.name == "schedstat":
            raise PermissionError()
        if path.name == "scaling_cur_freq":
            raise FileNotFoundError()
        return original(path)

    monkeypatch.setattr(perf_diagnostics.Path, "read_text", read)
    sample = perf_diagnostics._process_sample(psutil.Process())
    assert sample["rss_bytes"] > 0
    assert sample["scheduler"]["schedstat"] == {"error": "PermissionError"}
    assert sample["scheduler"]["frequency_khz"] == {"error": "FileNotFoundError"}


def test_kernel_sample_does_not_retain_arguments(monkeypatch):
    def read(path):
        return (
            "16 0xsecret 0xprivate 0xaddress\n"
            if path.name == "syscall"
            else "futex_wait_queue\n"
        )

    monkeypatch.setattr(perf_diagnostics.Path, "read_text", read)
    assert perf_diagnostics._kernel_sample(123) == {
        "syscall": "16",
        "wchan": "futex_wait_queue",
    }


def test_restricted_kernel_sample_preserves_process_counters(monkeypatch):
    original = perf_diagnostics.Path.read_text

    def read(path):
        if path.name in ("syscall", "wchan"):
            raise PermissionError()
        return original(path)

    monkeypatch.setattr(perf_diagnostics.Path, "read_text", read)
    sample = perf_diagnostics._process_sample(psutil.Process())
    assert sample["rss_bytes"] > 0
    assert sample["kernel"] == {
        "syscall": {"error": "PermissionError"},
        "wchan": {"error": "PermissionError"},
    }


def test_live_child_kernel_sample():
    with subprocess.Popen(
        [sys.executable, "-c", "print('ready', flush=True); input()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            sample = perf_diagnostics._process_sample(psutil.Process(child.pid))
            assert sample["pid"] == child.pid
            assert set(sample["kernel"]) == {"syscall", "wchan"}
            for value in sample["kernel"].values():
                assert isinstance(value, dict) or value is None or " " not in value
        finally:
            child.communicate("\n", timeout=5)


def test_unavailable_gpu_field_is_explicit():
    def unsupported():
        raise perf_diagnostics.pynvml.NVMLError_NotSupported()

    assert perf_diagnostics._nvml_value(unsupported) == {
        "error": "NVMLError_NotSupported"
    }


def test_sampler_attributes_nvml_delays(monkeypatch, tmp_path):
    monkeypatch.setenv(perf_diagnostics._ROOT_ENV, str(tmp_path))
    diagnostics = perf_diagnostics.AttemptDiagnostics(1)
    clock = [0.0]
    monkeypatch.setattr(perf_diagnostics.time, "monotonic", lambda: clock[0])
    nvml = perf_diagnostics.pynvml
    monkeypatch.setattr(nvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(nvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(nvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(nvml, "nvmlSystemGetDriverVersion", lambda: "test")

    def owners(handle):
        clock[0] += 3.0
        return [SimpleNamespace(pid=os.getpid())]

    def gpu_sample(index, handle):
        clock[0] += 7.0
        diagnostics.stop.set()
        return {"nvml_index": index}

    monkeypatch.setattr(nvml, "nvmlDeviceGetComputeRunningProcesses", owners)
    monkeypatch.setattr(perf_diagnostics, "_gpu_sample", gpu_sample)
    diagnostics._sample(os.getpid())
    diagnostics.finish(0)
    sample = _events(diagnostics.directory / "resources.jsonl")[-1]
    assert sample["processes"]
    assert sample["sample_started_wall_time_ns"] <= sample["wall_time_ns"]
    assert sample["process_sample_seconds"] == 0
    assert sample["sample_seconds"] == 10
    assert sample["gpu_query_timings"] == [
        {
            "nvml_index": 0,
            "ownership_seconds": 3,
            "metrics_seconds": 7,
            "total_seconds": 10,
        }
    ]


def test_artifact_failure_does_not_replace_test_status(monkeypatch, tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("not a directory")
    monkeypatch.setenv(perf_diagnostics._ROOT_ENV, str(occupied))
    diagnostics = perf_diagnostics.AttemptDiagnostics(1)
    assert diagnostics.directory is None
    diagnostics.finish(1)
    monkeypatch.setenv(perf_diagnostics._ATTEMPT_ENV, str(occupied))
    perf_diagnostics.record_request({"e2e_ms": 100})


def test_attempt_artifacts_keep_failures_and_real_request_producer(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(perf_diagnostics._ROOT_ENV, str(tmp_path / "diagnostics"))
    monkeypatch.chdir(tmp_path)
    # the real request producer flushes before the child fails; no pytest
    # session-finish hook is needed to preserve the failed attempt
    program = tmp_path / "request.py"
    program.write_text(
        """
import sys
import time
from sglang.multimodal_gen.test.server.test_server_common import DiffusionServerBase
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams, DiffusionServerArgs, DiffusionTestCase, PerformanceSummary,
)
runner = DiffusionServerBase()
runner._perf_results = []
case = DiffusionTestCase("probe", DiffusionServerArgs("test", modality="image"), DiffusionSamplingParams(prompt="test"))
print("BEGIN diffusion testcase: probe", flush=True)
for index in (1, 2):
    runner._record_performance_result(case, PerformanceSummary(float(sys.argv[1]), 5, 5, {}, [], {}, {}), index)
time.sleep(1.2)
sys.exit(int(sys.argv[2]))
"""
    )
    for attempt, (e2e, expected_rc) in enumerate(((100, 1), (50, 0)), 1):
        rc, output = pytest_runner._run_pytest_attempt(
            [sys.executable, str(program), str(e2e), str(expected_rc)], attempt=attempt
        )
        assert rc == expected_rc
        assert "BEGIN diffusion testcase: probe" in output
    directories = sorted((tmp_path / "diagnostics").iterdir())
    assert len(directories) == 2
    for directory, expected_e2e, expected_rc in zip(directories, (100, 50), (1, 0)):
        records = _events(directory / "requests.jsonl")
        assert [r["result"]["e2e_ms"] for r in records] == [expected_e2e] * 2
        assert [r["result"]["request_index"] for r in records] == [1, 2]
        resources = _events(directory / "resources.jsonl")
        assert any(r["event"] == "resources" and r["processes"] for r in resources)
        events = _events(directory / "events.jsonl")
        assert events[-1]["returncode"] == expected_rc
        assert events[-1]["sampler_stopped"] is True


def test_infrastructure_retries_keep_attempt_numbers(monkeypatch):
    seen = []

    def run(cmd, attempt):
        seen.append((cmd, attempt))
        if attempt == 1:
            return 1, "short test summary info\nFAILED case TimeoutError\n=== end ==="
        return 0, "passed"

    monkeypatch.setattr(pytest_runner, "_run_pytest_attempt", run)
    assert pytest_runner.run_pytest(["test.py"])[0] == 0
    assert [attempt for _, attempt in seen] == [1, 2]
    assert "--last-failed" not in seen[0][0]
    assert "--last-failed" in seen[1][0]
