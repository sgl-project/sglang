"""Evidence export controls on the independent native scheduler fixture."""

import pytest
from sglang_simulator.simulation.sglang import scheduler
from test_native_scheduler_hooks import hooked_scheduler as hooked_scheduler
from test_native_scheduler_hooks import mock_profile_output


@pytest.mark.parametrize(
    "level, compression",
    [
        ("full", "none"),
        ("full", "gzip"),
        ("full", "zstd"),
        ("metrics", "none"),
        ("metrics", "zstd"),
    ],
)
def test_profile_retention_preserves_metrics_and_request_export(
    hooked_scheduler, monkeypatch, tmp_path, level, compression
):
    import gzip
    import io
    import json

    from sglang_simulator.simulation.iteration_stats import IterationStats

    if compression == "zstd":
        pytest.importorskip("zstandard")
    test = hooked_scheduler
    mock_profile_output(monkeypatch)
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", level)
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_COMPRESSION", compression)
    monkeypatch.setattr(
        scheduler.Envs, "output_dir", classmethod(lambda cls: str(tmp_path))
    )
    monkeypatch.setattr(
        scheduler, "calc_metrics", lambda stats: {"request_count": len(stats)}
    )
    monkeypatch.setattr(
        scheduler,
        "calc_iteration_metrics",
        lambda rows, metrics: {
            "forward_total": sum(row["forward_latency"] for row in rows)
        },
    )
    stats = scheduler.request_stats_manager.get_req_stats("request")
    stats.input_length = 10
    stats.created_time = stats.queue_start = stats.queue_end = 100
    stats.last_event_time = 101
    stats.gen_token_latencies = [1]
    row = {
        "forward_latency": 0.25,
        "requests": [[1, 10]],
        "nested_evidence": {"id": "kept-in-full"},
    }
    store = IterationStats(
        lambda: tmp_path,
        compression="zstd" if compression == "zstd" else "zlib",
        metrics_only=level == "metrics",
    )
    store.append(row)
    monkeypatch.setattr(test.hook, "ITERATION_STATS", store)
    profile = next(iter(test.target._request_dispatcher._mapping.values()))
    assert profile(test.profile_req).success is True
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["request_count"] == 1 and metrics["forward_total"] == 0.25
    assert json.loads((tmp_path / "request.jsonl").read_text())["rid"] == "request"
    files = list(tmp_path.glob("iteration.jsonl*"))
    if level == "metrics":
        assert not files
        assert metrics["iteration_evidence"] == {
            "retention_policy": "metrics",
            "persisted": False,
            "omitted_by_policy": True,
        }
    else:
        assert len(files) == 1
        if compression == "gzip":
            raw = gzip.decompress(files[0].read_bytes())
        elif compression == "zstd":
            import zstandard

            with zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(files[0].read_bytes())
            ) as stream:
                raw = stream.read()
        else:
            raw = files[0].read_bytes()
        assert raw == (json.dumps(row) + "\n").encode()
        assert "iteration_evidence" not in metrics
    assert store.path is None  # Successful export clears only its own working spool.
