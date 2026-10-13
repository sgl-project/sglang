import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.utils import perf_logger as perf_logger_module
from sglang.multimodal_gen.runtime.utils.perf_logger import PerformanceLogger


def _metrics(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        stages={"DenoisingStage": 1.0},
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
