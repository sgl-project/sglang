import os
import subprocess
import sys
import textwrap

import pytest

from sglang.multimodal_gen.test.runner.pytest_runner import _is_retryable_failure


@pytest.mark.parametrize(
    "output",
    [
        "[performance] Validation failed for 'E2E Latency'",
        "[performance] Validation failed for 'E2E Latency'\nTimeoutError",
        "[performance] Validation failed for 'E2E Latency'\nCUDA out of memory",
        "multimodal_gen/test/server/test_server_utils.py: AssertionError",
        "Consistency check failed for example\nTimeoutError",
    ],
)
def test_validation_failures_are_not_retryable(output):
    assert not _is_retryable_failure(output)


@pytest.mark.parametrize(
    "output", ["TimeoutError", "SafetensorError", "CUDA out of memory"]
)
def test_infrastructure_failure_policy_is_unchanged(output):
    assert _is_retryable_failure(output)


@pytest.mark.parametrize(
    "problem", ["regression", "missing_baseline", "missing_record", "missing_e2e"]
)
def test_performance_failure_survives_real_pytest_runner(tmp_path, problem):
    # exercise the validator, request loop, pytest output and retry classifier together
    test_file = tmp_path / "test_guard.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest
            from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
            from sglang.multimodal_gen.test.server import test_server_common as common
            from sglang.multimodal_gen.test.server.testcase_configs import (
                DiffusionSamplingParams, DiffusionServerArgs, DiffusionTestCase, ScenarioConfig,
            )

            @pytest.mark.parametrize("case_id", ["threshold_guard"])
            def test_guard(case_id, monkeypatch):
                server = common.DiffusionServerBase()
                server._perf_results = []
                case = DiffusionTestCase(
                    case_id,
                    DiffusionServerArgs(model_path="test", modality="image", lora_path="test-lora"),
                    DiffusionSamplingParams(prompt="test"),
                    run_lora_basic_api_check=True, perf_repeat_requests=2,
                    run_consistency_check=False, run_models_api_check=False,
                    run_perf_check=PROBLEM not in ("missing_record", "missing_e2e"),
                )
                scenario = ScenarioConfig({}, {}, 1000, 100, 100)
                if PROBLEM == "missing_baseline":
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
                monkeypatch.setattr(server, "run_and_collect", collect)
                monkeypatch.setattr(
                    server, "_test_lora_api_functionality",
                    lambda *args: lora_checks.append(1),
                )
                try:
                    server._test_diffusion_generation_impl(case, None)
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
    env.pop("SGLANG_DIFFUSION_DIAGNOSTICS_DIR", None)
    env.pop("SGLANG_DIFFUSION_DIAGNOSTICS_ATTEMPT_DIR", None)
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
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "[performance]" in output, output
    assert "GUARD_REQUESTS=1 LORA_CHECKS=0" in output, output
    retained = 0 if problem in {"missing_record", "missing_e2e"} else 1
    assert f"RETAINED_METRICS={retained}" in output, output
    assert output.count("Starting pytest attempt") == 1, output
    assert "'threshold_guard': 'fail'" in output, output
