import json

from asc_bench.report import render_report
from asc_bench.sla import Row


def make_row(hash_, mean, rank=None, accuracy=0.91, tp=2):
    row = Row(cell_hash=hash_, cell_ids=[f"cell-{hash_}"], repeats=1)
    row.mean = dict(mean)
    row.std = {k: 0.0 for k in mean}
    row.rank = rank
    row.sla_pass = True
    row.accuracy = accuracy
    row.accuracy_ok = True
    row.tp_size = tp
    return row


def test_render_report_md_and_json(tmp_path):
    good = make_row("aaa", {"output_throughput": 1000.0, "p99_ttft_ms": 800.0}, rank=1)
    slow = make_row("bbb", {"output_throughput": 2000.0, "p99_ttft_ms": 3000.0})
    slow.sla_pass = False
    slow.rank = None
    rows = [slow, good]
    records = [
        {
            "cell_id": "cell-aaa",
            "cell_hash": "aaa",
            "status": "done",
            "metrics": good.mean,
            "gsm8k_accuracy": 0.91,
            "detail": None,
        },
        {
            "cell_id": "cell-bbb",
            "cell_hash": "bbb",
            "status": "failed_health_timeout",
            "metrics": None,
            "gsm8k_accuracy": None,
            "detail": "health timeout after 1800s",
        },
    ]
    provenance = {"sglang_version": "0.5.19", "git_sha": "6f481ad"}

    render_report(
        tmp_path,
        "unit-cfg",
        "unit-run-1",
        rows,
        records,
        provenance,
        {"p99_ttft_ms": 2000.0},
    )

    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Ranking" in md
    assert "cell-aaa" in md
    assert "failed_health_timeout" in md
    assert "SLA pass/fail matrix" in md
    assert "Compatibility matrix" in md
    assert "0.5.19" in md

    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["schema"] == 1
    assert payload["summary"] == {
        "total": 2,
        "done": 1,
        "failed": {"failed_health_timeout": 1},
    }
    by_rank = {r["rank"]: r for r in payload["ranked"]}
    assert by_rank[1]["cell_hash"] == "aaa"
    assert by_rank[None]["cell_hash"] == "bbb"
    assert by_rank[1]["output_throughput_per_card"] == 500.0  # tp=2
