import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.utils import perf_logger as perf_logger_module
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    RequestMetrics,
)


def _metrics(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        stages={"DenoisingStage": 1.0},
        denoising_stages={"DenoisingStage"},
        steps=[1.0],
        memory_snapshots={},
        total_duration_ms=1.0,
    )


def test_reader_during_a_dump_sees_the_previous_complete_report(tmp_path, monkeypatch):
    # clients poll this path; writing it in place exposed a truncated file
    path = tmp_path / "perf.json"
    monkeypatch.setattr(perf_logger_module, "get_git_commit_hash", lambda: "test")
    PerformanceLogger.dump_benchmark_report(str(path), _metrics("first"))

    seen_mid_write = []
    real_dump = json.dump

    def dump_while_reading(obj, fp, **kwargs):
        seen_mid_write.append(path.read_text())
        real_dump(obj, fp, **kwargs)

    monkeypatch.setattr(perf_logger_module.json, "dump", dump_while_reading)
    PerformanceLogger.dump_benchmark_report(str(path), _metrics("second"))

    assert [json.loads(text)["request_id"] for text in seen_mid_write] == ["first"]
    assert json.loads(path.read_text())["request_id"] == "second"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["perf.json"]


@pytest.mark.parametrize("is_output_rank", [True, False])
def test_only_the_output_rank_writes_the_report(tmp_path, monkeypatch, is_output_rank):
    monkeypatch.setattr(perf_logger_module, "get_git_commit_hash", lambda: "test")
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = is_output_rank
    worker.server_args = SimpleNamespace(model_path="model")
    path = tmp_path / "perf.json"

    worker._dump_perf_report(
        SimpleNamespace(perf_dump_path=str(path), is_warmup=False),
        SimpleNamespace(metrics=_metrics("request")),
    )

    assert path.exists() == is_output_rank


@pytest.mark.parametrize("is_output_rank", [True, False])
@pytest.mark.parametrize("case", ["request", "warmup", "no_path"])
def test_finalize_preserves_perf_report_ownership(
    tmp_path, monkeypatch, is_output_rank, case
):
    monkeypatch.setattr(perf_logger_module, "get_git_commit_hash", lambda: "test")
    monkeypatch.setattr(perf_logger_module, "get_is_main_process", lambda: False)
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = is_output_rank
    worker.server_args = SimpleNamespace(model_path="model")
    worker._record_output_peak_memory = lambda *args, **kwargs: None
    worker._record_replica_peak_memory = lambda *args: None
    path = tmp_path / "perf.json"
    req = SimpleNamespace(
        request_id="request",
        perf_dump_path=None if case == "no_path" else str(path),
        is_warmup=case == "warmup",
        suppress_logs=True,
        return_raw_frames=True,
    )
    output = OutputBatch(metrics=_metrics("request"))

    worker._finalize_output_batch(
        output_batch=output,
        req=req,
        save_output_paths=lambda batch: None,
        output_metrics=[],
        deferred=False,
    )

    assert path.exists() == (is_output_rank and case == "request")
    if path.exists():
        report = json.loads(path.read_text())
        assert report["request_id"] == "request"
        assert report["meta"] == {"model": "model"}


@pytest.mark.parametrize("is_output_rank", [True, False])
def test_finalize_preserves_the_output_rank_report_guard(
    tmp_path, monkeypatch, is_output_rank
):
    monkeypatch.setattr(perf_logger_module, "get_git_commit_hash", lambda: "test")
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = is_output_rank
    worker.server_args = SimpleNamespace(model_path="model")
    worker._materialize_output_transport = Mock()
    worker._record_output_peak_memory = Mock()
    worker._record_replica_peak_memory = Mock()
    path = tmp_path / "perf.json"
    metrics = RequestMetrics("request")

    worker._finalize_output_batch(
        output_batch=OutputBatch(metrics=metrics),
        req=SimpleNamespace(
            request_id="request",
            perf_dump_path=str(path),
            is_warmup=False,
            suppress_logs=True,
            return_raw_frames=True,
        ),
        save_output_paths=Mock(),
        output_metrics=[metrics],
        deferred=False,
    )

    worker._record_replica_peak_memory.assert_called_once_with([metrics])
    assert path.exists() == is_output_rank
    if is_output_rank:
        assert json.loads(path.read_text())["request_id"] == "request"
