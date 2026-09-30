import os
import subprocess
import sys
import textwrap
import time

import pytest

from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.runner.pytest_runner import (
    _estimate_failed_test_time,
    _is_retryable_failure,
    run_pytest,
)
from sglang.multimodal_gen.test.server import test_server_common as common
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    ScenarioConfig,
)


@pytest.mark.parametrize("generate_baseline", [False, True])
def test_e2e_only_does_not_require_stage_metrics(monkeypatch, generate_baseline):
    monkeypatch.setenv("SGLANG_GEN_BASELINE", str(int(generate_baseline)))
    case = DiffusionTestCase(
        "e2e_only",
        DiffusionServerArgs(model_path="test", modality="image"),
        DiffusionSamplingParams(prompt="test"),
        run_perf_check=False,
    )
    scenario = ScenarioConfig({}, {}, 1000, 0, 0, expected_load_ms=100)
    monkeypatch.setitem(common.BASELINE_CONFIG.scenarios, case.id, scenario)
    monkeypatch.setattr(common, "_PENDING_BASELINE_DUMPS", {})
    server = common.DiffusionServerBase()
    server._perf_results = []
    record = RequestPerfRecord(
        request_id="guard",
        commit_hash="test",
        tag="guard",
        stages=[],
        steps=[],
        total_duration_ms=2000 if generate_baseline else 1000,
    )
    server._validate_and_record(case, record, load_time_ms=100)
    assert len(server._perf_results) == 1
    assert bool(common._PENDING_BASELINE_DUMPS) == generate_baseline


@pytest.mark.parametrize(
    "output",
    [
        "multimodal_gen/test/server/test_server_utils.py: AssertionError",
        "Consistency check failed for example\nTimeoutError",
        "[performance] Validation failed\nConsistency check failed for example",
    ],
)
def test_validation_failures_are_not_retryable(output):
    assert not _is_retryable_failure(output)


@pytest.mark.parametrize(
    "output",
    [
        "[performance] Validation failed for 'E2E Latency'",
        "[performance] Validation failed for 'Load Latency (excluding warmup)'",
        "[performance] Validation failed for 'Average Denoise Step'\nTimeoutError",
        "[performance] E2E missing or invalid\nCUDA out of memory",
    ],
)
def test_performance_failures_are_retryable(output):
    assert _is_retryable_failure(output)


@pytest.mark.parametrize(
    "output", ["TimeoutError", "SafetensorError", "CUDA out of memory"]
)
def test_infrastructure_failure_policy_is_unchanged(output):
    assert _is_retryable_failure(output)


@pytest.mark.parametrize("with_deadline", [False, True])
def test_performance_retry_recovers_only_failed_items(
    tmp_path, monkeypatch, with_deadline
):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if with_deadline:
        monkeypatch.setenv("SGLANG_DIFFUSION_RETRY_DEADLINE", str(time.time() + 600))
    else:
        monkeypatch.delenv("SGLANG_DIFFUSION_RETRY_DEADLINE", raising=False)
    test_file = tmp_path / "test_retry.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "def test_slow():\n"
        "    marker = Path(__file__).with_suffix('.attempt')\n"
        "    if not marker.exists():\n"
        "        marker.touch()\n"
        "        assert False, '[performance] Validation failed for E2E Latency'\n"
        "def test_fast():\n"
        "    marker = Path(__file__).with_suffix('.passed')\n"
        "    assert not marker.exists(), 'passing case must not rerun'\n"
        "    marker.touch()\n"
    )
    code, _, _ = run_pytest([str(test_file)])
    assert code == 0
    assert test_file.with_suffix(".attempt").exists()
    assert test_file.with_suffix(".passed").exists()


def test_retry_budget_preserves_failure_and_report(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("SGLANG_DIFFUSION_RETRY_DEADLINE", str(time.time() - 1))
    test_file = tmp_path / "test_budget.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('case_id', ['slow_case'])\n"
        "def test_slow(case_id):\n"
        "    assert False, '[performance] Validation failed for E2E Latency'\n"
    )
    report = tmp_path / "junit.xml"
    code, executed, results = run_pytest([str(test_file)], junit_xml_path=str(report))
    output = capfd.readouterr().out
    assert code == 1
    assert executed == ["slow_case"]
    assert results == {"slow_case": "fail"}
    assert output.count("Starting pytest attempt") == 1
    assert "Retry budget exhausted" in output
    assert "Pytest Tail Summary" in output


def test_retry_estimate_excludes_successful_cases(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        "<testsuites><testsuite>"
        '<testcase name="passed" time="100" />'
        '<testcase name="failed" time="10"><failure /></testcase>'
        '<testcase name="error" time="20"><error /></testcase>'
        "</testsuite></testsuites>"
    )
    assert _estimate_failed_test_time(str(report), 130) == 30
    assert _estimate_failed_test_time(None, 130) == 130


@pytest.mark.parametrize(
    "problem",
    [
        "regression",
        "missing_baseline",
        "missing_record",
        "missing_e2e",
        "missing_log",
        "e2e_only_regression",
        "e2e_only_missing_baseline",
        "e2e_only_zero_baseline",
        "e2e_only_nan_baseline",
    ],
)
def test_performance_failure_survives_real_pytest_runner(tmp_path, problem):
    # exercise the validator, request loop, pytest output and retry classifier together
    test_file = tmp_path / "test_guard.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest
            from types import SimpleNamespace
            from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
            from sglang.multimodal_gen.test.server import test_server_common as common
            from sglang.multimodal_gen.test.test_utils import wait_for_req_perf_record
            from sglang.multimodal_gen.test.server.testcase_configs import (
                DiffusionSamplingParams, DiffusionServerArgs, DiffusionTestCase, ScenarioConfig,
            )

            @pytest.mark.parametrize("case_id", ["threshold_guard"])
            def test_guard(case_id, monkeypatch, tmp_path):
                server = common.DiffusionServerBase()
                server._perf_results = []
                case = DiffusionTestCase(
                    case_id,
                    DiffusionServerArgs(model_path="test", modality="image", lora_path="test-lora"),
                    DiffusionSamplingParams(prompt="test"),
                    run_lora_basic_api_check=True, perf_repeat_requests=2,
                    run_consistency_check=False, run_models_api_check=False,
                    run_perf_check=not PROBLEM.startswith("e2e_only_") and PROBLEM not in ("missing_record", "missing_e2e", "missing_log"),
                )
                scenario = ScenarioConfig({}, {}, 1000, 100, 100, expected_load_ms=100)
                if PROBLEM == "e2e_only_zero_baseline":
                    scenario.expected_e2e_ms = 0
                if PROBLEM == "e2e_only_nan_baseline":
                    scenario.expected_e2e_ms = float("nan")
                if PROBLEM in ("missing_baseline", "e2e_only_missing_baseline"):
                    monkeypatch.delitem(common.BASELINE_CONFIG.scenarios, case_id, raising=False)
                else:
                    monkeypatch.setitem(common.BASELINE_CONFIG.scenarios, case_id, scenario)
                monkeypatch.setattr(common.current_platform, "is_cuda", lambda: False)
                monkeypatch.setattr(common.current_platform, "is_hip", lambda: False)
                monkeypatch.setattr(common, "get_generate_fn", lambda **kwargs: None)
                requests = []
                lora_checks = []
                def collect(*args, **kwargs):
                    requests.append(1)
                    if PROBLEM == "missing_record":
                        return None, b""
                    return RequestPerfRecord(
                        request_id="guard", commit_hash="test", tag="guard",
                        stages=[], steps=[100],
                        total_duration_ms=None if PROBLEM == "missing_e2e" else 2000,
                    ), b""
                context = SimpleNamespace(load_time_ms=100)
                if PROBLEM == "missing_log":
                    log_path = tmp_path / "empty-perf.jsonl"
                    log_path.write_text("")
                    context = SimpleNamespace(perf_log_path=log_path, load_time_ms=100)
                    monkeypatch.setattr(server, "_client", lambda ctx: None)
                    def generate(*args):
                        requests.append(1)
                        return "guard", b""
                    monkeypatch.setattr(server, "_run_generation_with_server_watchdog", generate)
                    monkeypatch.setattr(
                        common, "wait_for_req_perf_record",
                        lambda rid, path, timeout: wait_for_req_perf_record(rid, path, timeout=0.01),
                    )
                else:
                    monkeypatch.setattr(server, "run_and_collect", collect)
                monkeypatch.setattr(
                    server, "_test_lora_api_functionality",
                    lambda *args: lora_checks.append(1),
                )
                try:
                    server._test_diffusion_generation_impl(case, context)
                finally:
                    print(f"GUARD_REQUESTS={len(requests)} LORA_CHECKS={len(lora_checks)}")
                    print(f"RETAINED_METRICS={len(server._perf_results)}")

            def test_unrelated_timeout():
                raise TimeoutError("independent infrastructure failure")
            """
        ).replace("PROBLEM", repr(problem))
    )
    report = tmp_path / "junit.xml"
    env = os.environ.copy()
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        SGLANG_GEN_BASELINE="0",
        SGLANG_GEN_GT="0",
    )
    command = (
        "from sglang.multimodal_gen.test.runner.pytest_runner import run_pytest; "
        f"result = run_pytest([{str(test_file)!r}], junit_xml_path={str(report)!r}); "
        "print('GUARD_RESULT', result); raise SystemExit(result[0])"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "[performance]" in output, output
    assert "GUARD_REQUESTS=1 LORA_CHECKS=0" in output, output
    retained = 0 if problem in {"missing_record", "missing_e2e", "missing_log"} else 1
    assert f"RETAINED_METRICS={retained}" in output, output
    assert output.count("Starting pytest attempt") == 7, output
    assert "Max retry exceeded (6)" in output, output
    assert "'threshold_guard': 'fail'" in output, output
