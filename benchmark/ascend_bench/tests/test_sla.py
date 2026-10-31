from asc_bench.sla import Row, aggregate_by_hash, evaluate, parse_last_metrics


def make_row(hash_, means, repeats=1, accuracy=None):
    row = Row(cell_hash=hash_, cell_ids=[f"c-{hash_}"], repeats=repeats)
    row.mean = dict(means)
    row.std = {k: 0.0 for k in means}
    row.accuracy = accuracy
    return row


def test_parse_last_metrics_takes_last_record(tmp_path):
    path = tmp_path / "bench.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"output_throughput": 100, "p99_ttft_ms": 900}',
                "not json",
                '{"output_throughput": 200, "p99_ttft_ms": 800}',
            ]
        ),
        encoding="utf-8",
    )
    metrics = parse_last_metrics(path)
    assert metrics["output_throughput"] == 200
    assert metrics["p99_ttft_ms"] == 800


def test_parse_last_metrics_missing_file(tmp_path):
    assert parse_last_metrics(tmp_path / "nope.jsonl") is None


def test_sla_threshold_filters_and_ranks():
    sla = type("S", (), {"thresholds": {"p99_ttft_ms": 1000}, "cv_max": 0.15})()
    fast = make_row("aaa", {"output_throughput": 300, "p99_ttft_ms": 500})
    slow = make_row("bbb", {"output_throughput": 900, "p99_ttft_ms": 1500})
    rows = {"aaa": fast, "bbb": slow}
    out = evaluate(rows, sla)
    assert fast.sla_pass is True
    assert slow.sla_pass is False
    assert fast.rank == 1 and slow.rank is None
    assert [r.cell_hash for r in out][0] == "aaa"


def test_variance_gate_marks_unrankable():
    sla = type("S", (), {"thresholds": {}, "cv_max": 0.15})()
    row = Row(cell_hash="noisy", cell_ids=["a", "b"], repeats=2)
    row.mean = {"output_throughput": 100.0}
    row.std = {"output_throughput": 30.0}
    rows = {"noisy": row}
    evaluate(rows, sla)
    assert row.unrankable is True
    assert row.rank is None
    assert "cv=" in row.unrankable_reason


def test_accuracy_floor_excludes_from_ranking():
    sla = type("S", (), {"thresholds": {}, "cv_max": 0.15})()
    good = make_row("good", {"output_throughput": 100}, accuracy=0.9)
    bad = make_row("bad", {"output_throughput": 500}, accuracy=0.1)
    evaluate({"good": good, "bad": bad}, sla, accuracy_floor=0.30)
    assert good.accuracy_ok is True
    assert bad.accuracy_ok is False
    assert good.rank == 1
    assert bad.rank is None


def test_aggregate_groups_by_hash_and_computes_std():
    per_cell = {
        "a0": {"output_throughput": 100.0, "p99_ttft_ms": 500.0},
        "a1": {"output_throughput": 200.0, "p99_ttft_ms": 700.0},
    }
    rows = aggregate_by_hash(per_cell, {"a0": "h", "a1": "h"}, {}, {})
    row = rows["h"]
    assert row.repeats == 2
    assert row.mean["output_throughput"] == 150.0
    assert abs(row.std["output_throughput"] - 70.71) < 0.01


def test_non_finite_metrics_are_skipped():
    """request_rate: Infinity in real bench JSONL must not crash stdev."""
    per_cell = {
        "a0": {"output_throughput": 100.0, "request_rate": float("inf")},
        "a1": {"output_throughput": 200.0, "request_rate": float("inf")},
    }
    rows = aggregate_by_hash(per_cell, {"a0": "h", "a1": "h"}, {}, {})
    row = rows["h"]
    assert row.mean["output_throughput"] == 150.0
    assert "request_rate" not in row.mean
