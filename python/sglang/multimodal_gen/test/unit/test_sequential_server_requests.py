import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server import conftest, test_server_common
from sglang.multimodal_gen.test.server.testcase_configs import (
    BaselineConfig,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    ScenarioConfig,
    ToleranceConfig,
)


def _perf_record():
    return RequestPerfRecord(
        request_id="request",
        commit_hash="test",
        tag="test",
        stages=[{"name": "DenoisingStage", "execution_time_ms": 10}],
        steps=[5, 5],
        total_duration_ms=100,
        memory_snapshots={
            "load_peak": {"peak_reserved_mb": 1000},
            "runtime_peak": {"peak_reserved_mb": 2000},
        },
    )


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setenv("SGLANG_GEN_GT", "0")
    monkeypatch.setenv("SGLANG_GEN_BASELINE", "0")
    monkeypatch.setenv("SGLANG_SKIP_CONSISTENCY", "0")
    monkeypatch.setattr(test_server_common.current_platform, "is_cuda", lambda: True)
    scenario = ScenarioConfig(
        stages_ms={"DenoisingStage": 10},
        denoise_step_ms={0: 5, 1: 5},
        expected_e2e_ms=100,
        expected_avg_denoise_ms=5,
        expected_median_denoise_ms=5,
        estimated_full_test_time_s=1,
        load_peak_vram_mb=1000,
        runtime_peak_vram_mb=2000,
    )
    monkeypatch.setattr(
        test_server_common,
        "BASELINE_CONFIG",
        BaselineConfig(
            scenarios={"first": scenario, "second": scenario},
            step_fractions=[0, 1],
            tolerances=ToleranceConfig(0, 0, 0, 0, 0),
            improvement_threshold=0,
        ),
    )
    runner = test_server_common.DiffusionServerBase()
    monkeypatch.setattr(type(runner), "_perf_results", [])
    monkeypatch.setattr(test_server_common, "_PENDING_BASELINE_DUMPS", {})
    monkeypatch.setattr(test_server_common, "get_generate_fn", Mock())
    monkeypatch.setattr(runner, "_validate_consistency", Mock())
    monkeypatch.setattr(runner, "_dump_baseline_for_testcase", Mock())
    case = DiffusionTestCase(
        "first",
        DiffusionServerArgs(model_path="test", modality="image"),
        DiffusionSamplingParams(prompt="first"),
        perf_repeat_requests=2,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    )
    return runner, case


@pytest.mark.parametrize("bad_request", [0, 1])
@pytest.mark.parametrize(
    "failure", ["performance", "load_peak", "runtime_peak", "missing_memory", "consistency", "generation"]
)
def test_each_request_failure_fails_case(harness, monkeypatch, bad_request, failure):
    runner, case = harness
    records = [_perf_record(), _perf_record()]
    outputs = [(record, b"output") for record in records]
    if failure == "performance":
        records[bad_request].total_duration_ms = 10000
    elif failure in {"load_peak", "runtime_peak"}:
        records[bad_request].memory_snapshots[failure]["peak_reserved_mb"] = 10000
    elif failure == "missing_memory":
        records[bad_request].memory_snapshots.clear()
    elif failure == "consistency":
        checks = [None, None]
        checks[bad_request] = AssertionError("wrong pixels or audio")
        runner._validate_consistency.side_effect = checks
    else:
        outputs[bad_request] = RuntimeError("server request failed")
    generate = Mock(side_effect=outputs)
    monkeypatch.setattr(runner, "run_and_collect", generate)
    ctx = object()

    with pytest.raises(pytest.fail.Exception, match=f"request {bad_request + 1}/2"):
        runner.test_diffusion_generation(case, ctx)

    assert generate.call_count == 2
    assert all(call.args[0] is ctx for call in generate.call_args_list)
    assert runner._validate_consistency.call_count == (1 if failure == "generation" else 2)
    # Even failed performance measurements must survive in the report.
    expected = [i + 1 for i in range(2) if failure != "generation" or i != bad_request]
    assert [r["request_index"] for r in runner._perf_results] == expected


def test_both_requests_pass(harness, monkeypatch):
    runner, case = harness
    monkeypatch.setattr(runner, "run_and_collect", Mock(side_effect=[(_perf_record(), b"first"), (_perf_record(), b"second")]))
    runner.test_diffusion_generation(case, object())
    assert runner._validate_consistency.call_count == 2
    assert [call.args[1] for call in runner._validate_consistency.call_args_list] == [b"first", b"second"]
    assert [r["request_index"] for r in runner._perf_results] == [1, 2]


def test_later_skip_cannot_hide_earlier_failure(harness, monkeypatch):
    runner, case = harness
    monkeypatch.setattr(runner, "run_and_collect", Mock(side_effect=[RuntimeError("failed first"), pytest.skip.Exception("skip second")]))
    with pytest.raises(pytest.fail.Exception, match="failed first"):
        runner.test_diffusion_generation(case, object())


def test_empty_content_is_not_a_consistency_pass(harness):
    runner, case = harness
    with pytest.raises(pytest.fail.Exception, match="Empty output"):
        test_server_common.DiffusionServerBase._validate_consistency(runner, case, b"")


@pytest.mark.parametrize("repeat", [0, -1])
def test_invalid_repeat_count(harness, repeat):
    _, case = harness
    with pytest.raises(ValueError, match="must be positive"):
        replace(case, perf_repeat_requests=repeat)


def test_report_keeps_both_requests_and_replaces_retry(tmp_path):
    path = tmp_path / "results.json"
    records = [
        {"class_name": "suite", "test_name": "case", "request_index": i, "e2e_ms": i}
        for i in (1, 2)
    ]
    conftest._write_results_json(records, str(path))
    conftest._write_results_json([{**records[0], "e2e_ms": 3}], str(path))
    assert json.loads(path.read_text()) == [{**records[0], "e2e_ms": 3}, records[1]]


def test_gt_generation_runs_both_requests(harness, monkeypatch):
    runner, case = harness
    monkeypatch.setenv("SGLANG_GEN_GT", "1")
    generate = Mock(side_effect=[(None, b"first"), (None, b"second")])
    monkeypatch.setattr(runner, "run_and_collect", generate)
    save = Mock()
    monkeypatch.setattr(runner, "_save_gt_output", save)
    runner.test_diffusion_generation(case, object())
    assert [(call.args[0].id, call.args[1]) for call in save.call_args_list] == [("first", b"first"), ("first", b"second")]
    assert all(not call.kwargs["collect_perf"] for call in generate.call_args_list)
    runner._validate_consistency.assert_not_called()
