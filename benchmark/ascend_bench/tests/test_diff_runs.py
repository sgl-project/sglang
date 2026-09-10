import json

from asc_bench.diff_runs import diff_runs, render_compare_md
from asc_bench.runner import Manifest

BASE_METRICS = {"output_throughput": 100.0, "p99_ttft_ms": 500.0}


def seed_run(tmp_path, cell_id, cell_hash, metrics):
    run_dir = tmp_path / cell_id
    (run_dir / "cells" / cell_id).mkdir(parents=True)
    manifest = Manifest(run_dir / "manifest.jsonl")
    manifest.header({"run_id": cell_id})
    manifest.append(cell_id, cell_hash, "done", metrics=dict(metrics), accuracy=None)
    manifest.close()
    return run_dir


def test_diff_joins_by_hash_and_computes_delta(tmp_path):
    run_a = seed_run(tmp_path, "cellA", "h1", BASE_METRICS)
    run_b = seed_run(
        tmp_path, "cellB", "h1", {"output_throughput": 120.0, "p99_ttft_ms": 450.0}
    )

    diffs = diff_runs(run_a, run_b)
    assert len(diffs) == 1
    diff = diffs[0]
    assert diff["cell_hash"] == "h1"
    assert diff["delta"]["output_throughput"] == 20.0
    assert diff["delta_pct"]["output_throughput"] == 20.0
    assert diff["delta_pct"]["p99_ttft_ms"] == -10.0


def test_diff_skips_non_overlapping_hashes(tmp_path):
    run_a = seed_run(tmp_path, "cellA", "h1", BASE_METRICS)
    run_b = seed_run(tmp_path, "cellB", "h2", BASE_METRICS)
    assert diff_runs(run_a, run_b) == []


def test_diff_reads_bench_jsonl_when_manifest_has_no_metrics(tmp_path):
    run_a = seed_run(tmp_path, "cellA", "h1", BASE_METRICS)
    # run B: manifest entry without metrics, but bench.jsonl present
    run_b = tmp_path / "runB"
    (run_b / "cells" / "cellB").mkdir(parents=True)
    manifest = Manifest(run_b / "manifest.jsonl")
    manifest.append("cellB", "h1", "done", metrics=None)
    manifest.close()
    (run_b / "cells" / "cellB" / "bench.jsonl").write_text(
        json.dumps({"output_throughput": 50.0, "p99_ttft_ms": 600.0}) + "\n",
        encoding="utf-8",
    )

    diffs = diff_runs(run_a, run_b)
    assert diffs[0]["delta"]["output_throughput"] == -50.0


def test_render_compare_md_is_valid_table(tmp_path):
    run_a = seed_run(tmp_path, "cellA", "h1", BASE_METRICS)
    run_b = seed_run(tmp_path, "cellB", "h1", BASE_METRICS)
    diffs = diff_runs(run_a, run_b)
    md = render_compare_md(diffs, ["output_throughput"])
    assert md.startswith("# Regression diff")
    assert "`h1`" in md and "cellA" in md and "cellB" in md
