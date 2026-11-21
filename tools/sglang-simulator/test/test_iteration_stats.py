"""Retain exact verbose evidence while reducing only small metric summaries."""

import gzip
import io
import json
import tracemalloc

import pytest
from sglang_simulator.simulation.iteration_stats import IterationStats
from sglang_simulator.simulation.utils import calc_iteration_metrics


def test_roundtrip_bytes_metrics_order_and_reset(tmp_path):
    rows = [
        {"forward_latency": 0.25, "native_worker_event": {"profile": "x" * 100000}},
        {"forward_latency": 0.1, "cpu_overhead": 0.01, "requests": [["r", 1]]},
        {"l2_load_latency": None, "cpu_overhead": 0.02, "name": "非 ASCII"},
    ]
    store = IterationStats(lambda: tmp_path / "output")
    for row in rows:
        store.append(row)
    expected = "".join(json.dumps(row) + "\n" for row in rows)
    output = io.StringIO()
    store.write_jsonl(output)
    assert output.getvalue() == expected
    assert list(store) == rows
    assert store[-1] == rows[-1]
    assert calc_iteration_metrics(store.metric_rows()) == calc_iteration_metrics(rows)
    spool = store.path
    assert spool.is_file() and spool.parent == tmp_path / "output"
    store.clear()
    assert not spool.exists() and len(store) == 0
    store.append(rows[0])
    assert list(store) == rows[:1]
    store.clear()


def test_native_evidence_not_retained_on_heap(tmp_path):
    store = IterationStats(lambda: tmp_path)
    tracemalloc.start()
    try:
        for index in range(128):
            store.append(
                {"forward_latency": 0.01, "evidence": str(index) + "x" * 1000000}
            )
        current, peak = tracemalloc.get_traced_memory()
        assert current < 1024 * 1024
        assert peak < 12 * 1024 * 1024
        assert len(store) == 128
        assert all("evidence" not in row for row in store.metric_rows())
    finally:
        tracemalloc.stop()
        store.clear()


def test_export_failure_preserves_evidence_and_retry(tmp_path):
    store = IterationStats(lambda: tmp_path)
    store.append({"native_worker_event": {"event_committed": True}})

    class BrokenDestination:
        def write(self, _):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store.write_jsonl(BrokenDestination())
    assert store.path.exists() and len(store) == 1
    output = io.StringIO()
    store.write_jsonl(output)
    assert json.loads(output.getvalue())["native_worker_event"]["event_committed"]
    store.clear()


def test_compressed_profile_export_preserves_jsonl(tmp_path, monkeypatch):
    from sglang_simulator.simulation.manager.env import Envs

    rows = [{"forward_latency": 0.1, "evidence": "x" * 1000000} for _ in range(3)]
    store = IterationStats(lambda: tmp_path)
    for row in rows:
        store.append(row)
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_COMPRESSION", "gzip")
    assert Envs.evidence_compression() == "gzip"
    with gzip.open(
        tmp_path / "iteration.jsonl.gz",
        "wt",
        encoding="utf-8",
        compresslevel=1,
        newline="\n",
    ) as output:
        store.write_jsonl(output)
    assert (
        gzip.decompress((tmp_path / "iteration.jsonl.gz").read_bytes())
        == ("".join(json.dumps(row) + "\n" for row in rows)).encode()
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_COMPRESSION", "unknown")
    with pytest.raises(ValueError, match="none or gzip"):
        Envs.evidence_compression()
    store.clear()


def test_truncated_spool_does_not_silently_export(tmp_path):
    store = IterationStats(lambda: tmp_path)
    store.append({"forward_latency": 0.1})
    store._stream.truncate(9)
    with pytest.raises(OSError, match="truncated"):
        store.write_jsonl(io.StringIO())
    assert store.path.exists()
    store.clear()


def test_write_failure_stops_further_admission(tmp_path):
    store = IterationStats(lambda: tmp_path / "not-a-directory")
    (tmp_path / "not-a-directory").write_text("occupied")
    with pytest.raises(FileExistsError):
        store.append({"forward_latency": 0.1})
    assert len(store) == 0
    with pytest.raises(RuntimeError, match="previously failed"):
        store.append({"forward_latency": 0.1})
    with pytest.raises(RuntimeError, match="previously failed"):
        store.write_jsonl(io.StringIO())


def test_failed_unlink_can_be_retried_without_losing_spool(tmp_path, monkeypatch):
    store = IterationStats(lambda: tmp_path)
    store.append({"forward_latency": 0.1})
    path_type = type(store.path)
    original = path_type.unlink

    def fail(_):
        raise PermissionError("busy spool")

    monkeypatch.setattr(path_type, "unlink", fail)
    with pytest.raises(PermissionError, match="busy spool"):
        store.clear()
    assert store.path.exists() and len(store) == 1
    monkeypatch.setattr(path_type, "unlink", original)
    store.clear()
    assert store.path is None and len(store) == 0
