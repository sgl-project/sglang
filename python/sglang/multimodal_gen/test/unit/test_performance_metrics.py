import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.multimodal_gen.runtime.managers.gpu_worker as gpu_worker_module
import sglang.multimodal_gen.runtime.managers.memory_managers.component_manager as component_manager_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    WarmupPhasePeak,
)
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


@pytest.fixture(autouse=True)
def _isolate_global_component_residency_manager(monkeypatch):
    # Worker paths exercised here create the process-global residency manager
    # for their fake pipeline; leaving it behind hands its modules to later
    # tests (the worker prefers the global manager's placement modules).
    monkeypatch.setattr(
        component_manager_module, "_GLOBAL_COMPONENT_RESIDENCY_MANAGER", None
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


def test_request_metrics_attributes_steps_and_iterations_to_active_stage():
    metrics = RequestMetrics("request")
    metrics.active_stage_name = "ShapeStage"

    metrics.record_step(0.1)
    metrics.record_stage_iterations(4, 50)
    metrics.active_stage_name = "PaintStage"
    metrics.record_step(0.2)
    metrics.record_stage_iterations(4, 30)

    assert metrics.steps == [100.0, 200.0]
    assert metrics.steps_by_stage == {
        "ShapeStage": [100.0],
        "PaintStage": [200.0],
    }
    assert metrics.stage_iterations == {
        "ShapeStage": (4, 50),
        "PaintStage": (4, 30),
    }


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
    worker._warmup_peak_reserved_mb = 0.0
    output = OutputBatch()
    metrics = RequestMetrics("request")
    replica_group = Mock()
    replica_group.all_reduce.return_value = torch.tensor(
        [5120.0, 3584.0, 6144.0], dtype=torch.float64
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
    assert metrics.memory_snapshots["warmup_peak"].peak_reserved_mb == 6144.0


def test_server_warmup_preserves_peak_after_managed_stage_timeline():
    worker = GPUWorker.__new__(GPUWorker)
    worker._auto_residency_warmup_records = []
    residency_manager = Mock()
    residency_manager.take_warmup_phase_peaks.return_value = {
        "0:denoise:use:transformer": WarmupPhasePeak(("transformer",), 8)
    }
    residency_manager.current_device_components.return_value = ("transformer",)
    device_module = Mock()
    device_module.max_memory_allocated.return_value = 9
    device_module.max_memory_reserved.return_value = 12
    req = SimpleNamespace(
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=1,
        metrics=None,
    )

    with (
        patch.object(torch, "get_device_module", return_value=device_module),
        patch.object(
            gpu_worker_module,
            "peek_global_component_residency_manager",
            return_value=residency_manager,
        ),
    ):
        worker._record_server_warmup_memory(
            req=req,
            workload=(128, 96, 9, 2),
            baseline_allocated_bytes=3,
            succeeded=True,
        )

    record = worker._auto_residency_warmup_records[0]
    assert record.peak_allocated_bytes == 9
    assert record.peak_reserved_bytes == 12
    assert (record.width, record.height, record.num_frames) == (128, 96, 9)
    assert record.num_inference_steps == 2
    assert record.phase_peak_allocated_bytes["request:untracked"] == 9
    assert record.phase_active_components["request:untracked"] == ("transformer",)


def test_server_warmup_does_not_treat_allocator_cache_as_untracked_live_memory():
    worker = GPUWorker.__new__(GPUWorker)
    worker._auto_residency_warmup_records = []
    residency_manager = Mock()
    residency_manager.take_warmup_phase_peaks.return_value = {
        "0:denoise:use:transformer": WarmupPhasePeak(("transformer",), 8)
    }
    residency_manager.current_device_components.return_value = ("transformer",)
    device_module = Mock()
    device_module.max_memory_allocated.return_value = 8
    device_module.max_memory_reserved.return_value = 12
    req = SimpleNamespace(
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=1,
        metrics=None,
    )

    with (
        patch.object(torch, "get_device_module", return_value=device_module),
        patch.object(
            gpu_worker_module,
            "peek_global_component_residency_manager",
            return_value=residency_manager,
        ),
    ):
        worker._record_server_warmup_memory(
            req=req,
            workload=(64, 64, 1, 1),
            baseline_allocated_bytes=3,
            succeeded=True,
        )

    record = worker._auto_residency_warmup_records[0]
    assert "request:untracked" not in record.phase_peak_allocated_bytes


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


def _warmup_batch_for_iterations(steps: int, target_steps: int):
    from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics

    metrics = RequestMetrics(request_id="probe")
    metrics.suppress_stage_breakdown = False
    metrics.active_stage_name = "DenoisingStage"
    return SimpleNamespace(
        metrics=metrics,
        num_inference_steps=steps,
        extra={"warmup_target_num_inference_steps": target_steps},
        is_warmup=True,
    )


def test_stage_formula_records_probe_and_default_iterations():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        record_default_workload_iterations,
    )

    batch = _warmup_batch_for_iterations(steps=4, target_steps=50)
    stage = SimpleNamespace(default_workload_iterations=lambda batch, steps: steps - 1)
    record_default_workload_iterations(stage, batch)
    assert batch.metrics.stage_iterations == {"DenoisingStage": (3, 49)}


def test_fixed_schedule_formula_records_the_same_count_twice():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        record_default_workload_iterations,
    )

    batch = _warmup_batch_for_iterations(steps=4, target_steps=50)
    stage = SimpleNamespace(default_workload_iterations=lambda batch, steps: 8)
    record_default_workload_iterations(stage, batch)
    assert batch.metrics.stage_iterations == {"DenoisingStage": (8, 8)}


def test_explicit_loop_record_wins_over_the_formula():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        record_default_workload_iterations,
    )

    batch = _warmup_batch_for_iterations(steps=4, target_steps=50)
    batch.metrics.record_stage_iterations(12, 12)
    stage = SimpleNamespace(default_workload_iterations=lambda batch, steps: steps)
    record_default_workload_iterations(stage, batch)
    assert batch.metrics.stage_iterations == {"DenoisingStage": (12, 12)}


def test_stage_without_a_formula_records_nothing():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        record_default_workload_iterations,
    )

    batch = _warmup_batch_for_iterations(steps=4, target_steps=50)
    stage = SimpleNamespace(default_workload_iterations=lambda batch, steps: None)
    record_default_workload_iterations(stage, batch)
    assert batch.metrics.stage_iterations == {}


def test_warmup_peak_stays_out_of_the_runtime_peak():
    from unittest import mock

    worker = gpu_worker_module.GPUWorker.__new__(gpu_worker_module.GPUWorker)
    worker._runtime_peak_reserved_mb = 0.0
    worker._warmup_peak_reserved_mb = 0.0
    worker.is_output_rank = False
    peaks = iter([40000.0, 20000.0])
    with (
        mock.patch.object(
            gpu_worker_module.current_platform, "is_cpu", return_value=False
        ),
        mock.patch.object(
            gpu_worker_module,
            "capture_memory_snapshot",
            side_effect=lambda: SimpleNamespace(peak_reserved_mb=next(peaks)),
        ),
    ):
        worker._record_output_peak_memory(SimpleNamespace(), is_warmup=True)
        worker._record_output_peak_memory(SimpleNamespace(), is_warmup=False)
    assert worker._warmup_peak_reserved_mb == 40000.0
    assert worker._runtime_peak_reserved_mb == 20000.0


def _warmup_req(*, probe: bool = False) -> SimpleNamespace:
    extra = {"auto_residency_full_shape_probe": True} if probe else {}
    return SimpleNamespace(is_warmup=True, extra=extra)


def test_worker_releases_the_probe_pool_before_the_next_request(monkeypatch):
    calls = []
    fake_device = SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        gpu_worker_module.torch, "get_device_module", lambda: fake_device
    )
    monkeypatch.setattr(type(current_platform), "is_cpu", lambda self: False)
    monkeypatch.setattr(type(current_platform), "is_mps", lambda self: False)
    worker = GPUWorker.__new__(GPUWorker)
    worker._release_warmup_pool_before_serving = False

    worker._release_warmup_pool(_warmup_req())
    worker._release_warmup_pool(_warmup_req(probe=True))
    assert calls == []

    # the bounded re-warm after the probe regrows the pool from empty
    worker._release_warmup_pool(_warmup_req())
    assert calls == ["empty_cache"]

    worker._release_warmup_pool(SimpleNamespace(is_warmup=False, extra={}))
    assert calls == ["empty_cache"]


def test_worker_keeps_the_pool_when_no_probe_ran(monkeypatch):
    calls = []
    fake_device = SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        gpu_worker_module.torch, "get_device_module", lambda: fake_device
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker._release_warmup_pool_before_serving = False

    worker._release_warmup_pool(_warmup_req())
    worker._release_warmup_pool(SimpleNamespace(is_warmup=False, extra={}))
    assert calls == []
