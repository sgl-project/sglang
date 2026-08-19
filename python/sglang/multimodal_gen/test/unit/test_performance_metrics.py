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


def test_performance_summary_uses_largest_peak_reserved_memory():
    summary = PerformanceSummary.from_req_perf_record(
        _perf_record(
            {
                "after_forward": {"peak_reserved_mb": 1024.0},
                "lifetime_peak": {"peak_reserved_mb": 4096.0},
            }
        ),
        step_fractions=(),
    )

    assert summary.peak_vram_mb == 4096.0


def test_worker_records_replica_lifetime_peak():
    worker = GPUWorker.__new__(GPUWorker)
    worker.local_rank = 0
    worker.is_output_rank = True
    worker._lifetime_peak_reserved_mb = 4096.0
    output = OutputBatch()
    metrics = RequestMetrics("request")
    replica_group = Mock()
    replica_group.all_reduce.return_value = torch.tensor(5120.0, dtype=torch.float64)
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
        worker._record_lifetime_peak_memory([metrics])

    assert output.peak_memory_mb == 3072.0
    assert worker._lifetime_peak_reserved_mb == 4096.0
    assert metrics.memory_snapshots["lifetime_peak"].peak_reserved_mb == 5120.0


def test_baseline_config_loads_peak_vram(tmp_path):
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
                        "peak_vram": 0.04,
                    }
                },
                "sampling": {"step_fractions": [0.0, 1.0]},
                "scenarios": {},
                "peak_vram_mb": {"case": 1234.5},
            }
        ),
        encoding="utf-8",
    )

    config = BaselineConfig.load(path)

    assert config.peak_vram_mb == {"case": 1234.5}
    assert config.tolerances.peak_vram == 0.04


def test_peak_vram_validation_uses_relative_tolerance():
    validator = PerformanceValidator(
        scenario=ScenarioConfig({}, {}, 0.0, 0.0, 0.0),
        tolerances=ToleranceConfig(0.0, 0.0, 0.0, 0.0, 0.0, peak_vram=0.05),
        step_fractions=(),
    )
    summary = PerformanceSummary(0.0, 0.0, 0.0, {}, [], {}, {}, 10_501.0)

    with patch.object(current_platform, "is_hip", return_value=False):
        with pytest.raises(AssertionError, match="Lifetime Peak VRAM"):
            validator.validate_peak_vram(summary, expected_peak_vram_mb=10_000.0)


def test_results_json_merges_retry_sessions(tmp_path):
    path = tmp_path / "diffusion-results.json"
    _write_results_json(
        [{"class_name": "Suite", "test_name": "first", "peak_vram_mb": 1.0}],
        str(path),
    )
    _write_results_json(
        [{"class_name": "Suite", "test_name": "second", "peak_vram_mb": 2.0}],
        str(path),
    )

    results = json.loads(path.read_text(encoding="utf-8"))
    assert {item["test_name"] for item in results} == {"first", "second"}
