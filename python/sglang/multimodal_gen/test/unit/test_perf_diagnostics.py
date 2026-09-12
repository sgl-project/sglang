import json
import os
import sys

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


def test_unavailable_gpu_field_is_explicit():
    def unsupported():
        raise perf_diagnostics.pynvml.NVMLError_NotSupported()

    assert perf_diagnostics._nvml_value(unsupported) == {
        "error": "NVMLError_NotSupported"
    }


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


def test_retry_policy_and_attempt_numbers_are_unchanged(monkeypatch):
    seen = []

    def run(cmd, attempt):
        seen.append((cmd, attempt))
        if attempt == 1:
            return 1, "short test summary info\nFAILED case [performance]\n=== end ==="
        return 0, "passed"

    monkeypatch.setattr(pytest_runner, "_run_pytest_attempt", run)
    assert pytest_runner.run_pytest(["test.py"])[0] == 0
    assert [attempt for _, attempt in seen] == [1, 2]
    assert "--last-failed" not in seen[0][0]
    assert "--last-failed" in seen[1][0]
