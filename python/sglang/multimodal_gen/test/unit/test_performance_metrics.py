import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.multimodal_gen.runtime.managers.gpu_worker as gpu_worker_module
import sglang.multimodal_gen.runtime.managers.memory_managers.component_manager as component_manager_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    GIB_BYTES,
    DefaultWorkload,
    WarmupMemoryRecord,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    WarmupPhasePeak,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
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
def _discrete_host_for_worker_tests(monkeypatch):
    # The worker's budget is the device's free memory on a discrete GPU; on a
    # shared host/device pool it is the pool's available share instead (see
    # test_shared_memory_pool). These tests describe the discrete case.
    monkeypatch.setattr(
        type(current_platform),
        "device_shares_host_memory",
        classmethod(lambda cls: False),
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
                "load_peak": {"peak_reserved_mb": 4096.0, "peak_allocated_mb": 3900.0},
                "runtime_peak": {
                    "peak_reserved_mb": 3072.0,
                    "peak_allocated_mb": 2900.0,
                },
            }
        ),
        step_fractions=(),
    )

    assert summary.load_peak_vram_mb == 4096.0
    assert summary.runtime_peak_vram_mb == 3072.0
    assert summary.load_peak_allocated_mb == 3900.0
    assert summary.runtime_peak_allocated_mb == 2900.0


def test_worker_records_replica_load_and_runtime_peaks():
    worker = GPUWorker.__new__(GPUWorker)
    worker.local_rank = 0
    worker.is_output_rank = True
    worker._load_peak_reserved_mb = 4096.0
    worker._runtime_peak_reserved_mb = 0.0
    worker._warmup_peak_reserved_mb = 0.0
    worker._load_peak_allocated_mb = 3000.0
    worker._runtime_peak_allocated_mb = 0.0
    output = OutputBatch()
    metrics = RequestMetrics("request")
    replica_group = Mock()
    replica_group.all_reduce.return_value = torch.tensor(
        [5120.0, 3584.0, 6144.0, 3500.0, 2560.0], dtype=torch.float64
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
    assert worker._runtime_peak_allocated_mb == 2048.0
    assert metrics.memory_snapshots["load_peak"].peak_reserved_mb == 5120.0
    assert metrics.memory_snapshots["runtime_peak"].peak_reserved_mb == 3584.0
    assert metrics.memory_snapshots["load_peak"].peak_allocated_mb == 3500.0
    assert metrics.memory_snapshots["runtime_peak"].peak_allocated_mb == 2560.0
    assert metrics.memory_snapshots["warmup_peak"].peak_reserved_mb == 6144.0


def test_server_warmup_preserves_peak_after_managed_stage_timeline():
    worker = GPUWorker.__new__(GPUWorker)
    worker._auto_residency_warmup_records = []
    worker.pipeline = SimpleNamespace(stages=[])
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
            baseline_reserved_bytes=3,
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
    worker.pipeline = SimpleNamespace(stages=[])
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
            baseline_reserved_bytes=12,
            succeeded=True,
        )

    record = worker._auto_residency_warmup_records[0]
    assert "request:untracked" not in record.phase_peak_allocated_bytes


def test_server_warmup_normalizes_profile_stage_names_for_residency_timing():
    worker = GPUWorker.__new__(GPUWorker)
    worker._auto_residency_warmup_records = []
    stage = Mock()
    stage._active_profile_stage_name.return_value = "RefinerStage"
    stage._component_stage_name.return_value = "refiner"
    worker.pipeline = SimpleNamespace(stages=[stage])
    residency_manager = Mock()
    residency_manager.take_warmup_phase_peaks.return_value = {
        "0:refiner:use:transformer_2": WarmupPhasePeak(
            ("transformer_2",),
            8,
            used_components=("transformer_2",),
        )
    }
    residency_manager.current_device_components.return_value = ("transformer_2",)
    device_module = Mock()
    device_module.max_memory_allocated.return_value = 8
    device_module.max_memory_reserved.return_value = 8
    req = SimpleNamespace(
        metrics=SimpleNamespace(
            total_duration_ms=12.0,
            stages={"RefinerStage": 10.0},
            steps=[],
            steps_by_stage={},
            stage_iterations={},
        )
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
            baseline_reserved_bytes=3,
            succeeded=True,
        )

    record = worker._auto_residency_warmup_records[0]
    assert record.stage_duration_ms == {"refiner": 10.0}
    _, stage_duration_ns, component_stages = (
        gpu_worker_module.estimate_default_workload_timing(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=1,
        )
    )
    assert stage_duration_ns == {"refiner": 10_000_000}
    assert component_stages == {"transformer_2": ("refiner",)}


def test_loaded_prequantized_checkpoint_can_use_auto_residency():
    worker = GPUWorker.__new__(GPUWorker)
    worker.rank = 0
    worker.is_output_rank = True
    worker._auto_residency_budget_correction_bytes = 0
    worker._auto_residency_reference_request_duration_ns = None
    worker._auto_residency_reference_stage_duration_ns = {}
    worker._auto_residency_reference_component_stages = {}
    quantized_module = torch.nn.Module()
    quantized_module.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(1, dtype=torch.int8), requires_grad=False),
    )
    worker.pipeline = SimpleNamespace(
        modules={"transformer": quantized_module},
        stages=[],
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    host_pin_budget = HostPinBudget(available_bytes=0, reserve_bytes=0)
    worker.server_args = SimpleNamespace(
        num_gpus=1,
        nnodes=1,
        node_rank=0,
        residency_mode=lambda _name: "resident",
        configured_residency_mode=lambda _name: "resident",
        explicit_residency_mode=lambda _name: None,
        component_residency_requirement=lambda _name: None,
        auto_residency_mode=lambda _name: None,
        ltx2_two_stage_device_mode=None,
        layerwise_tuning_for=lambda _name, *, dit_group: (0.0, 0.0, "leading"),
        is_layerwise_residency_policy_explicit=lambda _name, *, dit_group: False,
        pin_cpu_memory=True,
        host_pin_budget=lambda: host_pin_budget,
        component_quantizations=(),
        quantization=None,
        direct_gpu_weight_loading=False,
        nunchaku_config=None,
    )
    device_module = Mock()
    device_module.mem_get_info.return_value = (100, 100)
    device_module.memory_reserved.return_value = 0
    device_module.memory_allocated.return_value = 1

    with patch.object(torch, "get_device_module", return_value=device_module):
        report = worker._build_auto_residency_report(
            workload=DefaultWorkload(
                width=64,
                height=64,
                num_frames=1,
                num_inference_steps=1,
            ),
            records=[
                WarmupMemoryRecord(
                    width=64,
                    height=64,
                    num_frames=1,
                    baseline_allocated_bytes=1,
                    peak_allocated_bytes=2,
                    succeeded=True,
                )
            ],
        )

    assert report.skip_reason is None


def test_auto_residency_budget_respects_test_device_memory_cap(monkeypatch):
    worker = GPUWorker.__new__(GPUWorker)
    worker.rank = 0
    worker.is_output_rank = True
    worker._auto_residency_reference_request_duration_ns = None
    worker._auto_residency_reference_stage_duration_ns = {}
    worker._auto_residency_reference_component_stages = {}
    worker.pipeline = SimpleNamespace(
        modules={"transformer": torch.nn.Linear(1, 1)},
        stages=[],
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    host_pin_budget = HostPinBudget(available_bytes=0, reserve_bytes=0)
    worker.server_args = SimpleNamespace(
        num_gpus=1,
        nnodes=1,
        node_rank=0,
        residency_mode=lambda _name: "resident",
        configured_residency_mode=lambda _name: "resident",
        explicit_residency_mode=lambda _name: None,
        component_residency_requirement=lambda _name: None,
        auto_residency_mode=lambda _name: None,
        ltx2_two_stage_device_mode=None,
        layerwise_tuning_for=lambda _name, *, dit_group: (0.0, 0.0, "leading"),
        is_layerwise_residency_policy_explicit=lambda _name, *, dit_group: False,
        pin_cpu_memory=True,
        host_pin_budget=lambda: host_pin_budget,
        component_quantizations=(),
        quantization=None,
        direct_gpu_weight_loading=False,
        nunchaku_config=None,
    )
    device_module = Mock()
    device_module.mem_get_info.return_value = (100 * GIB_BYTES, 140 * GIB_BYTES)
    device_module.memory_reserved.return_value = 2 * GIB_BYTES
    device_module.memory_allocated.return_value = 60 * GIB_BYTES
    monkeypatch.setattr(
        gpu_worker_module.envs,
        "SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB",
        80,
    )

    with patch.object(torch, "get_device_module", return_value=device_module):
        report = worker._build_auto_residency_report(
            workload=DefaultWorkload(
                width=64,
                height=64,
                num_frames=1,
                num_inference_steps=1,
            ),
            records=[
                WarmupMemoryRecord(
                    width=64,
                    height=64,
                    num_frames=1,
                    baseline_allocated_bytes=1,
                    peak_allocated_bytes=60 * GIB_BYTES,
                    peak_reserved_bytes=62 * GIB_BYTES,
                    succeeded=True,
                )
            ],
        )

    assert report.budget_bytes == 80 * GIB_BYTES
    assert report.planning_headroom_correction_bytes == 2 * GIB_BYTES
    assert report.device_transition_allocated_bytes == 60 * GIB_BYTES


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
                        "runtime_peak_allocated_mb": 2000.5,
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
    assert scenario.load_peak_allocated_mb is None
    assert scenario.runtime_peak_allocated_mb == 2000.5
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


def test_peak_vram_validation_enforces_allocated_when_baselined():
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
    # reserved drifts with allocator pool history: +5% is not a regression
    # once the allocated peak is baselined and unchanged
    reserved_drift = PerformanceSummary(
        0.0,
        0.0,
        0.0,
        {},
        [],
        {},
        {},
        load_peak_vram_mb=10_000.0,
        runtime_peak_vram_mb=10_500.0,
        load_peak_allocated_mb=8_000.0,
        runtime_peak_allocated_mb=8_000.0,
    )
    allocated_regression = PerformanceSummary(
        0.0,
        0.0,
        0.0,
        {},
        [],
        {},
        {},
        load_peak_vram_mb=10_000.0,
        runtime_peak_vram_mb=10_000.0,
        load_peak_allocated_mb=8_000.0,
        runtime_peak_allocated_mb=8_300.0,
    )

    with patch.object(current_platform, "is_hip", return_value=False):
        validator.validate_peak_vram(
            reserved_drift,
            10_000.0,
            10_000.0,
            expected_runtime_peak_allocated_mb=8_000.0,
        )
        with pytest.raises(AssertionError, match=r"Runtime Peak VRAM \(allocated\)"):
            validator.validate_peak_vram(
                allocated_regression,
                10_000.0,
                10_000.0,
                expected_runtime_peak_allocated_mb=8_000.0,
            )
        # without an allocated baseline the reserved figure is still enforced
        with pytest.raises(AssertionError, match="Runtime Peak VRAM"):
            validator.validate_peak_vram(reserved_drift, 10_000.0, 10_000.0)

        reserved_drift.warmup_peak_vram_mb = 12_000.0
        with pytest.raises(AssertionError, match="Warmup Peak VRAM"):
            validator.validate_peak_vram(
                reserved_drift,
                10_000.0,
                10_000.0,
                expected_warmup_peak_vram_mb=10_000.0,
                expected_runtime_peak_allocated_mb=8_000.0,
            )


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
    worker._runtime_peak_allocated_mb = 0.0
    worker._warmup_peak_reserved_mb = 0.0
    worker.is_output_rank = False
    with (
        mock.patch.object(
            gpu_worker_module.current_platform, "is_cpu", return_value=False
        ),
        mock.patch.object(
            gpu_worker_module,
            "capture_memory_snapshot",
            side_effect=[
                MemorySnapshot(0.0, 0.0, 35000.0, 40000.0),
                MemorySnapshot(0.0, 0.0, 15000.0, 20000.0),
            ],
        ),
    ):
        worker._record_output_peak_memory(SimpleNamespace(), is_warmup=True)
        worker._record_output_peak_memory(SimpleNamespace(), is_warmup=False)
    assert worker._warmup_peak_reserved_mb == 40000.0
    assert worker._runtime_peak_reserved_mb == 20000.0
    assert worker._runtime_peak_allocated_mb == 15000.0


def _warmup_req(*, probe: bool = False) -> SimpleNamespace:
    extra = {"auto_residency_full_shape_probe": True} if probe else {}
    return SimpleNamespace(is_warmup=True, extra=extra)


def test_worker_releases_the_probe_pool_before_the_next_request(monkeypatch):
    import sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a as ipc_a2a_module

    calls = []
    fake_device = SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        gpu_worker_module.torch, "get_device_module", lambda: fake_device
    )
    monkeypatch.setattr(type(current_platform), "is_cpu", lambda self: False)
    monkeypatch.setattr(type(current_platform), "is_mps", lambda self: False)
    monkeypatch.setattr(
        ipc_a2a_module.IPC_A2A, "drop_staging", lambda: calls.append("drop_staging")
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker._release_warmup_pool_before_serving = False

    worker._release_warmup_pool(_warmup_req())
    worker._release_warmup_pool(_warmup_req(probe=True))
    assert calls == []

    # the bounded re-warm after the probe regrows the pool from empty, and the
    # IPC staging buffers sized for the probe's messages go with it
    worker._release_warmup_pool(_warmup_req())
    assert calls == ["drop_staging", "empty_cache"]

    worker._release_warmup_pool(SimpleNamespace(is_warmup=False, extra={}))
    assert calls == ["drop_staging", "empty_cache"]


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
