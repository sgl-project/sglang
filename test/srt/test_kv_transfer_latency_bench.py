import csv
import json
from pathlib import Path

from scripts.playground.disaggregation.kv_transfer_bench import kv_auto_experiment as auto
from scripts.playground.disaggregation.kv_transfer_bench import (
    kv_resweep_fine_dense_experiment as resweep,
)
from scripts.playground.disaggregation.kv_transfer_bench import kv_transfer_latency as bench
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


def test_parser_accepts_paced_transfer_options():
    args = build_parser().parse_args(
        [
            "--role",
            "initiator",
            "--host",
            "fd03:4514:80:6240::1",
            "--target-info-json",
            "{}",
            "--sizes",
            "512MB",
            "--rate-limit-gbps",
            "50",
            "--chunk-size",
            "8MB",
            "--background-duration-seconds",
            "120",
            "--background-bytes",
            "512MB",
            "--flow-id",
            "flow-a",
            "--start-at-unix-ns",
            "123456789",
        ]
    )

    assert args.rate_limit_gbps == 50.0
    assert args.chunk_size == "8MB"
    assert args.background_duration_seconds == 120.0
    assert args.background_bytes == "512MB"
    assert args.flow_id == "flow-a"
    assert args.start_at_unix_ns == 123456789


class _FakeEngine:
    def __init__(self, ret_values=None):
        self.calls = []
        self.ret_values = list(ret_values or [])

    def transfer_sync(self, session_id, src_ptr, dst_ptr, length):
        self.calls.append((session_id, src_ptr, dst_ptr, length))
        if self.ret_values:
            return self.ret_values.pop(0)
        return 0


class _FakeClock:
    def __init__(self):
        self.now_ns = 0
        self.sleeps = []

    def now(self):
        return self.now_ns

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now_ns += int(seconds * 1_000_000_000)


def test_transfer_sync_paced_preserves_single_transfer_when_unlimited():
    engine = _FakeEngine()
    target = TargetInfo(
        session_id="target:1234",
        host="target",
        gpu_id=0,
        ptr=2000,
        bytes=4096,
        ib_device="mlx5_bond_0",
        protocol="rdma",
    )
    clock = _FakeClock()

    ret = bench._transfer_sync_paced(
        engine,
        target,
        src_ptr=1000,
        num_bytes=300,
        chunk_size=128,
        rate_limit_gbps=None,
        now_ns_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    assert ret == 0
    assert engine.calls == [("target:1234", 1000, 2000, 300)]
    assert clock.sleeps == []


def test_transfer_sync_paced_chunks_and_sleeps_to_target_rate():
    engine = _FakeEngine()
    target = TargetInfo(
        session_id="target:1234",
        host="target",
        gpu_id=0,
        ptr=2000,
        bytes=4096,
        ib_device="mlx5_bond_0",
        protocol="rdma",
    )
    clock = _FakeClock()

    ret = bench._transfer_sync_paced(
        engine,
        target,
        src_ptr=1000,
        num_bytes=300,
        chunk_size=128,
        rate_limit_gbps=0.000008,
        now_ns_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    assert ret == 0
    assert engine.calls == [
        ("target:1234", 1000, 2000, 128),
        ("target:1234", 1128, 2128, 128),
        ("target:1234", 1256, 2256, 44),
    ]
    assert [round(v, 3) for v in clock.sleeps] == [0.128, 0.128, 0.044]


def test_transfer_sync_paced_stops_on_transfer_error():
    engine = _FakeEngine(ret_values=[0, -1, 0])
    target = TargetInfo(
        session_id="target:1234",
        host="target",
        gpu_id=0,
        ptr=2000,
        bytes=4096,
        ib_device="mlx5_bond_0",
        protocol="rdma",
    )
    clock = _FakeClock()

    ret = bench._transfer_sync_paced(
        engine,
        target,
        src_ptr=1000,
        num_bytes=300,
        chunk_size=128,
        rate_limit_gbps=0.000008,
        now_ns_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    assert ret == -1
    assert engine.calls == [
        ("target:1234", 1000, 2000, 128),
        ("target:1234", 1128, 2128, 128),
    ]


def test_wait_until_unix_ns_sleeps_until_deadline(monkeypatch):
    times = iter([1_000_000_000, 1_040_000_000, 1_060_000_000])
    sleeps = []

    monkeypatch.setattr(bench.time, "time_ns", lambda: next(times))
    monkeypatch.setattr(bench.time, "sleep", sleeps.append)

    bench._wait_until_unix_ns(1_050_000_000)

    assert sleeps == [0.05, 0.01]


def test_auto_parser_accepts_multi_hca_suites():
    args = auto.build_parser().parse_args(["--suite", "multi-hca-bg"])
    assert args.suite == "multi-hca-bg"

    args = auto.build_parser().parse_args(["--suite", "head-tcp-sweep"])
    assert args.suite == "head-tcp-sweep"

    args = auto.build_parser().parse_args(["--suite", "multi-hca-portcap-bg"])
    assert args.suite == "multi-hca-portcap-bg"

    args = auto.build_parser().parse_args(["--suite", "multi-hca-bgcap-only"])
    assert args.suite == "multi-hca-bgcap-only"

    args = auto.build_parser().parse_args(["--suite", "multi-hca-compare-4x100"])
    assert args.suite == "multi-hca-compare-4x100"

    args = auto.build_parser().parse_args(["--suite", "ratelimit-empty"])
    assert args.suite == "ratelimit-empty"


def test_resweep_parser_accepts_visible_suites():
    for suite in (
        "all-visible",
        "singleflow",
        "head-tcp-sweep",
        "legacy-multi-hca-unaverage",
        "small-flow",
        "competition",
    ):
        args = resweep.build_parser().parse_args(["--suite", suite])
        assert args.suite == suite


def test_resweep_fine_dense_sizes_are_logical_per_shard():
    one_shard = resweep.fine_dense_sizes_for_shards(1, auto.DENSE_SIZES_1).split(",")
    assert one_shard == auto.HEAD_TCP_FINE_DENSE_SIZES.split(",")

    two_shards = resweep.fine_dense_sizes_for_shards(2, auto.DENSE_SIZES_2).split(",")
    assert two_shards[:4] == ["512KB", "1MB", "1536KB", "2MB"]
    assert two_shards[31] == "16MB"
    assert two_shards[32] == "24MB"
    assert [auto.parse_size(size) * 2 for size in two_shards[:32]] == [
        size * 1024**2 for size in range(1, 33)
    ]

    four_shards = resweep.fine_dense_sizes_for_shards(4, auto.DENSE_SIZES_4).split(",")
    assert four_shards[:4] == ["256KB", "512KB", "768KB", "1MB"]
    assert four_shards[31] == "8MB"
    assert four_shards[32] == "12MB"
    assert [auto.parse_size(size) * 4 for size in four_shards[:32]] == [
        size * 1024**2 for size in range(1, 33)
    ]
    assert auto.parse_size(four_shards[-1]) * 4 == 2 * 1024**3


def test_resweep_all_visible_covers_86_experiments_without_duplicates():
    args = resweep.build_parser().parse_args(["--suite", "all-visible"])
    matrix_runs = resweep.selected_matrix_runs(args)
    competition_cases = resweep.selected_competition_cases(args)
    names = [run.run for run in matrix_runs] + [case.run for case in competition_cases]

    assert len(matrix_runs) == 68
    assert len(competition_cases) == 18
    assert len(names) == 86
    assert len(set(names)) == 86

    expected_prefix = [size * 1024**2 for size in range(1, 33)]
    for run in matrix_runs:
        logical_sizes = resweep.logical_sizes_bytes(run)
        assert logical_sizes[:32] == expected_prefix
        assert all(size > 32 * 1024**2 for size in logical_sizes[32:])


def test_resweep_small_flow_recreates_head_and_tail_profiles():
    args = resweep.build_parser().parse_args(["--suite", "small-flow"])
    runs = {run.run: run for run in resweep.small_flow_runs(args)}

    assert sorted(runs) == [
        "head_rdma_small_cap2x200_split",
        "head_rdma_small_cap4x100_split",
        "head_rdma_small_uncapped_4x200_split",
        "tail_rdma_small_cap100_single_nic",
        "tail_rdma_small_cap200_single_nic",
        "tail_rdma_small_uncapped_single_nic",
    ]

    head_uncapped = runs["head_rdma_small_uncapped_4x200_split"]
    assert head_uncapped.shards == 4
    assert len(head_uncapped.lanes) == 4
    assert head_uncapped.max_bytes == "512MB"
    assert head_uncapped.fg_rate_gbps is None
    assert head_uncapped.sizes.split(",")[:3] == ["256KB", "512KB", "768KB"]
    assert resweep.logical_sizes_bytes(head_uncapped)[0] == 1024**2
    assert resweep.logical_sizes_bytes(head_uncapped)[-1] == 2 * 1024**3

    tail_uncapped = runs["tail_rdma_small_uncapped_single_nic"]
    assert tail_uncapped.shards == 1
    assert tail_uncapped.max_bytes == "2GB"
    assert tail_uncapped.fg_rate_gbps is None
    assert tail_uncapped.lanes[0].ib_device == "mlx5_0"


def test_resweep_lane_host_overrides_update_rdma_defaults():
    args = resweep.build_parser().parse_args(
        [
            "--suite",
            "fixed-missing",
            "--src-rdma-hosts",
            "src0,src1,src2,src3",
            "--tgt-rdma-hosts",
            "tgt0,tgt1,tgt2,tgt3",
        ]
    )

    old_src = dict(auto.SRC_IPS)
    old_tgt = dict(auto.TGT_IPS)
    try:
        resweep.apply_lane_host_overrides(args)
        runs = {run.run: run for run in resweep.fixed_missing_runs(args)}
    finally:
        auto.SRC_IPS.clear()
        auto.SRC_IPS.update(old_src)
        auto.TGT_IPS.clear()
        auto.TGT_IPS.update(old_tgt)

    split = runs["800_4x200_bg1_cap200_moonbg_fixed"]
    assert [endpoint.src_host for endpoint in split.lanes] == ["src0", "src1", "src2", "src3"]
    assert [endpoint.tgt_host for endpoint in split.lanes] == ["tgt0", "tgt1", "tgt2", "tgt3"]

    single = runs["200_1x200_tail_single_nic_bg1_cap200_moonbg_fixed"]
    assert single.lanes[0].src_host == "src0"
    assert single.lanes[0].tgt_host == "tgt0"


def test_head_tcp_sweep_runs_cover_100g_200g_and_fine_sizes():
    args = auto.build_parser().parse_args(["--suite", "head-tcp-sweep"])
    runs = {run.run: run for run in auto.head_tcp_sweep_runs(args)}

    assert sorted(runs) == [
        "100_1x100_head_tcp_bg0_fine_dense",
        "200_1x200_head_tcp_bg0_fine_dense",
    ]

    fine_sizes = [f"{size}MB" for size in range(1, 33)]
    for rate, name in (
        (100.0, "100_1x100_head_tcp_bg0_fine_dense"),
        (200.0, "200_1x200_head_tcp_bg0_fine_dense"),
    ):
        run = runs[name]
        assert run.protocol == "tcp"
        assert run.shards == 1
        assert run.max_bytes == "2GB"
        assert run.lane_cap_gbps == rate
        assert run.bg_rate_gbps == 0.0
        assert run.fg_rate_gbps == rate
        assert run.capfill_lanes == ()
        assert run.sizes.split(",")[:32] == fine_sizes
        assert "48MB" in run.sizes.split(",")
        assert "2GB" in run.sizes.split(",")
        assert run.lanes[0] == auto.Endpoint(
            0,
            0,
            "192.168.0.39",
            "192.168.0.41",
            "mlx5_bond_0",
        )


def test_multi_hca_bg_uses_one_logical_flow_over_device_group():
    runs = auto.multi_hca_matrix_runs()
    run = next(r for r in runs if r.run == "200_2x100_bg50_multi_hca_portcap_moonbg")

    assert run.shards == 1
    assert run.max_bytes == "2GB"
    assert run.sizes == auto.DENSE_SIZES_1
    assert run.bg_rate_gbps == 100.0
    assert run.fg_rate_gbps == 100.0
    assert len(run.lanes) == 1
    assert run.lanes[0].ib_device == "mlx5_bond_0,mlx5_bond_1"
    assert run.capfill_rate_gbps == 100.0
    assert [endpoint.ib_device for endpoint in run.capfill_lanes] == [
        "mlx5_bond_0",
        "mlx5_bond_1",
    ]


def test_multi_hca_2x200_needs_no_port_cap_fillers():
    runs = auto.multi_hca_matrix_runs()
    run = next(r for r in runs if r.run == "400_2x200_bg50_multi_hca_portcap_moonbg")

    assert run.bg_rate_gbps == 200.0
    assert run.fg_rate_gbps == 200.0
    assert run.capfill_rate_gbps == 0.0
    assert run.capfill_lanes == ()


def test_bgcap_only_leaves_foreground_uncapped():
    runs = auto.multi_hca_bgcap_only_runs()
    run = next(r for r in runs if r.run == "400_4x100_bg50_multi_hca_bgcap_only")

    assert run.shards == 1
    assert run.max_bytes == "2GB"
    assert run.sizes == auto.DENSE_SIZES_1
    assert run.bg_rate_gbps == 200.0
    assert run.fg_rate_gbps is None
    assert run.capfill_rate_gbps == 0.0
    assert run.capfill_lanes == ()
    assert run.lanes[0].ib_device == "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"


def test_multi_hca_monitor_devices_expand_and_dedupe_groups():
    endpoints = [
        auto.Endpoint(0, 0, "src0", "tgt0", "mlx5_bond_0,mlx5_bond_1"),
        auto.Endpoint(1, 1, "src1", "tgt1", "mlx5_bond_1,mlx5_bond_2"),
    ]

    assert auto.monitor_ib_devices(endpoints) == [
        "mlx5_bond_0",
        "mlx5_bond_1",
        "mlx5_bond_2",
    ]


def test_4x100_compare_suite_contains_split_and_multi_hca_cases():
    runs = {run.run: run for run in auto.multi_hca_compare_4x100_runs()}

    split = runs["400_4x100_bg50_cap100_moonbg_split"]
    multi = runs["400_4x100_bg50_multi_hca_portcap_moonbg"]

    assert split.shards == 4
    assert len(split.lanes) == 4
    assert split.bg_rate_gbps == 50.0
    assert split.fg_rate_gbps == 50.0

    assert multi.shards == 1
    assert len(multi.lanes) == 1
    assert multi.bg_rate_gbps == 200.0
    assert multi.fg_rate_gbps == 200.0
    assert multi.capfill_rate_gbps == 100.0
    assert [endpoint.ib_device for endpoint in multi.capfill_lanes] == [
        "mlx5_bond_0",
        "mlx5_bond_1",
        "mlx5_bond_2",
        "mlx5_bond_3",
    ]
    assert multi.lanes[0].ib_device == "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"


def test_ratelimit_empty_suite_defines_split_and_multi_hca_bg0_baselines():
    runs = {run.run: run for run in auto.ratelimit_empty_runs()}

    assert sorted(runs) == [
        "200_1x200_bg0_ratelimit_multihca",
        "200_1x200_bg0_ratelimit_split",
        "400_2x200_bg0_ratelimit_multihca",
        "400_2x200_bg0_ratelimit_split",
        "400_4x100_bg0_ratelimit_multihca",
        "400_4x100_bg0_ratelimit_split",
        "800_4x200_bg0_ratelimit_multihca",
        "800_4x200_bg0_ratelimit_split",
    ]

    split_4x200 = runs["800_4x200_bg0_ratelimit_split"]
    assert split_4x200.shards == 4
    assert len(split_4x200.lanes) == 4
    assert split_4x200.max_bytes == "512MB"
    assert split_4x200.sizes == auto.DENSE_SIZES_4
    assert split_4x200.bg_rate_gbps == 0.0
    assert split_4x200.fg_rate_gbps == 200.0

    multi_4x200 = runs["800_4x200_bg0_ratelimit_multihca"]
    assert multi_4x200.shards == 1
    assert len(multi_4x200.lanes) == 1
    assert multi_4x200.lanes[0].ib_device == (
        "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
    )
    assert multi_4x200.max_bytes == "2GB"
    assert multi_4x200.sizes == auto.DENSE_SIZES_1
    assert multi_4x200.bg_rate_gbps == 0.0
    assert multi_4x200.fg_rate_gbps == 800.0

    split_4x100 = runs["400_4x100_bg0_ratelimit_split"]
    multi_4x100 = runs["400_4x100_bg0_ratelimit_multihca"]
    assert split_4x100.fg_rate_gbps == 100.0
    assert multi_4x100.fg_rate_gbps == 400.0

    split_2x200 = runs["400_2x200_bg0_ratelimit_split"]
    multi_2x200 = runs["400_2x200_bg0_ratelimit_multihca"]
    assert split_2x200.shards == 2
    assert split_2x200.max_bytes == "1GB"
    assert split_2x200.sizes == auto.DENSE_SIZES_2
    assert split_2x200.fg_rate_gbps == 200.0
    assert multi_2x200.fg_rate_gbps == 400.0

    split_1x200 = runs["200_1x200_bg0_ratelimit_split"]
    multi_1x200 = runs["200_1x200_bg0_ratelimit_multihca"]
    assert split_1x200.shards == 1
    assert split_1x200.fg_rate_gbps == 200.0
    assert multi_1x200.shards == 1
    assert multi_1x200.fg_rate_gbps == 200.0


def test_ratelimit_empty_runs_do_not_start_background_targets():
    args = auto.build_parser().parse_args(["--suite", "ratelimit-empty", "--dry-run"])
    runner = auto.Runner(args)
    run = next(
        r
        for r in auto.ratelimit_empty_runs()
        if r.run == "400_4x100_bg0_ratelimit_multihca"
    )

    script = runner.remote_target_script(run, list(run.lanes), Path("/tmp/bg0"), run.max_bytes)

    assert "target-bg" not in script
    assert "target-bond0" in script
