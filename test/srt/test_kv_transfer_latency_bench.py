import csv
import json

from scripts.playground.disaggregation.kv_transfer_bench.kv_transfer_latency import (
    TargetInfo,
    build_parser,
    format_bytes,
    load_target_info,
    parse_size,
    parse_size_list,
    summarize_latencies_ms,
    write_csv_summary,
    write_jsonl_samples,
    write_target_info,
)


def test_parse_size_accepts_binary_units():
    assert parse_size("1KB") == 1024
    assert parse_size("2MiB") == 2 * 1024 * 1024
    assert parse_size("3g") == 3 * 1024**3


def test_parse_size_list_expands_ranges_and_csv():
    assert parse_size_list("1MB,2MB") == [1024**2, 2 * 1024**2]
    assert parse_size_list("1MB:8MB:x2") == [
        1024**2,
        2 * 1024**2,
        4 * 1024**2,
        8 * 1024**2,
    ]


def test_summarize_latencies_ms_reports_percentiles_and_bandwidth():
    summary = summarize_latencies_ms([1.0, 2.0, 3.0, 4.0], num_bytes=1024**3)

    assert summary["latency_ms_mean"] == 2.5
    assert summary["latency_ms_p50"] == 2.5
    assert summary["latency_ms_p90"] == 3.7
    assert round(summary["bandwidth_GBps_p50"], 3) == 400.0


def test_format_bytes_uses_readable_units():
    assert format_bytes(1024) == "1.00KiB"
    assert format_bytes(1024**2) == "1.00MiB"


def test_target_info_roundtrip(tmp_path):
    path = tmp_path / "target.json"
    info = TargetInfo(
        session_id="192.168.0.42:12345",
        host="192.168.0.42",
        gpu_id=0,
        ptr=123456,
        bytes=1024,
        ib_device="mlx5_0",
        protocol="rdma",
    )

    write_target_info(path, info)

    assert load_target_info(path) == info


def test_result_writers_create_csv_and_jsonl(tmp_path):
    csv_path = tmp_path / "summary.csv"
    jsonl_path = tmp_path / "samples.jsonl"
    rows = [{"bytes": 1024, "latency_ms_p50": 1.5, "error_count": 0}]
    samples = [{"bytes": 1024, "iteration": 0, "latency_ms": 1.5, "ret": 0}]

    write_csv_summary(csv_path, rows)
    write_jsonl_samples(jsonl_path, samples)

    with csv_path.open() as f:
        assert list(csv.DictReader(f))[0]["bytes"] == "1024"
    with jsonl_path.open() as f:
        assert json.loads(f.readline())["latency_ms"] == 1.5


def test_parser_accepts_target_role():
    args = build_parser().parse_args(
        [
            "--role",
            "target",
            "--host",
            "192.168.0.42",
            "--max-bytes",
            "1GB",
            "--target-info-file",
            "/tmp/target.json",
        ]
    )

    assert args.role == "target"
    assert args.max_bytes == "1GB"


def test_parser_accepts_initiator_role():
    args = build_parser().parse_args(
        [
            "--role",
            "initiator",
            "--host",
            "192.168.0.41",
            "--target-info-file",
            "/tmp/target.json",
            "--sizes",
            "1MB:8MB:x2",
        ]
    )

    assert args.role == "initiator"
    assert args.sizes == "1MB:8MB:x2"
