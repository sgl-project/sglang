import json
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.multimodal_gen.runtime.managers.gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    MemorySnapshot,
    RequestMetrics,
    RequestPerfRecord,
)
from sglang.multimodal_gen.test.server.conftest import _write_results_json
from sglang.multimodal_gen.test.server.test_server_utils import PerformanceValidator
from sglang.multimodal_gen.test.server.testcase_configs import (
    BaselineConfig,
    PerformanceSummary,
    ScenarioConfig,
    ToleranceConfig,
)


def _perf_record(memory_snapshots: dict[str, dict]) -> RequestPerfRecord:
    return RequestPerfRecord(
        request_id="request",
        commit_hash="commit",
        tag="test",
        stages=[],
        steps=[],
        total_duration_ms=1.0,
        memory_snapshots=memory_snapshots,
    )


def test_performance_summary_separates_load_and_runtime_peaks():
    summary = PerformanceSummary.from_req_perf_record(
        _perf_record(
            {
                "after_forward": {"peak_reserved_mb": 1024.0},
                "load_peak": {"peak_reserved_mb": 4096.0},
                "runtime_peak": {"peak_reserved_mb": 3072.0},
            }
        ),
        step_fractions=(),
    )

    assert summary.load_peak_vram_mb == 4096.0
    assert summary.runtime_peak_vram_mb == 3072.0


def test_worker_records_replica_load_and_runtime_peaks():
    worker = GPUWorker.__new__(GPUWorker)
    worker.local_rank = 0
    worker.is_output_rank = True
    worker._load_peak_reserved_mb = 4096.0
    worker._runtime_peak_reserved_mb = 0.0
    output = OutputBatch()
    metrics = RequestMetrics("request")
    replica_group = Mock()
    replica_group.all_reduce.return_value = torch.tensor(
        [5120.0, 3584.0], dtype=torch.float64
    )
    snapshots = [
        MemorySnapshot(0.0, 0.0, 2048.0, 3072.0),
        MemorySnapshot(0.0, 0.0, 2048.0, 3072.0),
    ]

    with (
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device", return_value=torch.device("cpu")),
        patch.object(
            gpu_worker_module, "capture_memory_snapshot", side_effect=snapshots
        ),
        patch.object(
            gpu_worker_module, "get_replica_group", return_value=replica_group
        ),
    ):
        worker._record_output_peak_memory(output)
        worker._record_replica_peak_memory([metrics])

    assert output.peak_memory_mb == 3072.0
    assert worker._load_peak_reserved_mb == 4096.0
    assert worker._runtime_peak_reserved_mb == 3072.0
    assert metrics.memory_snapshots["load_peak"].peak_reserved_mb == 5120.0
    assert metrics.memory_snapshots["runtime_peak"].peak_reserved_mb == 3584.0


def test_baseline_config_loads_per_scenario_peak_vram(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "tolerances": {
                    "pr_test": {
                        "e2e": 0.25,
                        "denoise_stage": 0.25,
                        "non_denoise_stage": 0.8,
                        "denoise_step": 0.3,
                        "denoise_agg": 0.2,
                        "load_peak_vram": 0.01,
                        "runtime_peak_vram": 0.02,
                    }
                },
                "sampling": {"step_fractions": [0.0, 1.0]},
                "scenarios": {
                    "case": {
                        "stages_ms": {},
                        "denoise_step_ms": {},
                        "expected_e2e_ms": 1.0,
                        "expected_avg_denoise_ms": 1.0,
                        "expected_median_denoise_ms": 1.0,
                        "load_peak_vram_mb": 1234.5,
                        "runtime_peak_vram_mb": 2345.6,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = BaselineConfig.load(path)

    scenario = config.scenarios["case"]
    assert scenario.load_peak_vram_mb == 1234.5
    assert scenario.runtime_peak_vram_mb == 2345.6
    assert config.tolerances.load_peak_vram == 0.01
    assert config.tolerances.runtime_peak_vram == 0.02


def test_peak_vram_validation_uses_independent_tolerances():
    validator = PerformanceValidator(
        scenario=ScenarioConfig({}, {}, 0.0, 0.0, 0.0),
        tolerances=ToleranceConfig(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            load_peak_vram=0.01,
            runtime_peak_vram=0.02,
        ),
        step_fractions=(),
    )
    load_regression = PerformanceSummary(
        0.0,
        0.0,
        0.0,
        {},
        [],
        {},
        {},
        load_peak_vram_mb=10_129.0,
        runtime_peak_vram_mb=10_000.0,
    )
    runtime_regression = PerformanceSummary(
        0.0,
        0.0,
        0.0,
        {},
        [],
        {},
        {},
        load_peak_vram_mb=10_000.0,
        runtime_peak_vram_mb=10_201.0,
    )

    with patch.object(current_platform, "is_hip", return_value=False):
        with pytest.raises(AssertionError, match="Load Peak VRAM"):
            validator.validate_peak_vram(load_regression, 10_000.0, 10_000.0)
        with pytest.raises(AssertionError, match="Runtime Peak VRAM"):
            validator.validate_peak_vram(runtime_regression, 10_000.0, 10_000.0)


@pytest.mark.parametrize(
    ("load_peak_vram_mb", "runtime_peak_vram_mb", "message"),
    [
        (0.0, 10_000.0, "Load peak VRAM metric missing"),
        (10_000.0, 0.0, "Runtime peak VRAM metric missing"),
    ],
)
def test_peak_vram_validation_rejects_missing_metrics(
    load_peak_vram_mb, runtime_peak_vram_mb, message
):
    validator = PerformanceValidator(
        scenario=ScenarioConfig({}, {}, 0.0, 0.0, 0.0),
        tolerances=ToleranceConfig(0.0, 0.0, 0.0, 0.0, 0.0),
        step_fractions=(),
    )
    summary = PerformanceSummary(
        0.0,
        0.0,
        0.0,
        {},
        [],
        {},
        {},
        load_peak_vram_mb=load_peak_vram_mb,
        runtime_peak_vram_mb=runtime_peak_vram_mb,
    )

    with pytest.raises(AssertionError, match=message):
        validator.validate_peak_vram(summary, 10_000.0, 10_000.0)


def test_results_json_merges_retry_sessions(tmp_path):
    path = tmp_path / "diffusion-results.json"
    _write_results_json(
        [
            {
                "class_name": "Suite",
                "test_name": "first",
                "load_peak_vram_mb": 1.0,
                "runtime_peak_vram_mb": 2.0,
            }
        ],
        str(path),
    )
    _write_results_json(
        [
            {
                "class_name": "Suite",
                "test_name": "second",
                "load_peak_vram_mb": 3.0,
                "runtime_peak_vram_mb": 4.0,
            }
        ],
        str(path),
    )

    results = json.loads(path.read_text(encoding="utf-8"))
    assert {item["test_name"] for item in results} == {"first", "second"}
