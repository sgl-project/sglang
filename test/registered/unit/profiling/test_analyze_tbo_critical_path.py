import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


SCRIPT_DIR = (
    Path(__file__).resolve().parents[4]
    / ".claude"
    / "skills"
    / "llm-torch-profiler-analysis"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "analyze_tbo_critical_path", SCRIPT_DIR / "analyze_tbo_critical_path.py"
)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def gpu_event(name, ts, dur, stream, pid=8):
    return {
        "name": name,
        "cat": "kernel",
        "ph": "X",
        "pid": pid,
        "tid": stream,
        "ts": ts,
        "dur": dur,
        "args": {"stream": stream},
    }


def write_gzip_trace(path, events):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def test_interval_union_does_not_double_count():
    intervals = [(0, 10), (2, 5), (8, 15), (20, 21)]
    assert ANALYZER.merge_intervals(intervals) == [(0.0, 15.0), (20.0, 21.0)]
    assert ANALYZER.interval_union_us(intervals) == 16
    assert ANALYZER.interval_intersection_us(intervals, [(4, 12)]) == 8


def test_comm_hidden_and_exposed_from_gzip_trace(tmp_path):
    trace_path = tmp_path / "synthetic-TP-0.trace.json.gz"
    events = [
        gpu_event("compute", 0, 10, 0),
        gpu_event("ep_dispatch_intranode_0", 2, 6, 1),
        gpu_event("ep_combine_intranode_0", 8, 4, 1),
        gpu_event("ncclDevKernel_Generic", 20, 5, 2),
        # A second GPU pid must not contaminate the selected, heavier pid.
        gpu_event("compute", 0, 1, 0, pid=9),
        {
            "name": "hipStreamWaitEvent",
            "cat": "cuda_runtime",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 5,
            "dur": 3,
            "args": {},
        },
    ]
    write_gzip_trace(trace_path, events)

    result = ANALYZER.analyze_rank_trace(trace_path, rank=0)

    assert result["selected_gpu_pid"] == "8"
    assert result["communication"]["union_us"] == 15
    assert result["communication"]["hidden_by_other_stream_non_comm_us"] == 8
    assert result["communication"]["exposed_us"] == 7
    assert result["communication"]["phases"]["flydsl_dispatch"]["union_us"] == 6
    assert result["communication"]["phases"]["flydsl_combine"]["exposed_us"] == 2
    assert result["communication"]["phases"]["dp_nccl"]["exposed_us"] == 5
    window = result["forward_window_inference"]["window_metrics"][0]
    assert window["wall_span_us"] == 25
    assert window["busy_union_us"] == 17
    assert window["communication"]["union_us"] == 15
    assert window["communication"]["hidden_by_other_stream_non_comm_us"] == 8
    assert result["wait_events"]["total_count"] == 1
    assert result["wait_events"]["total_duration_us"] == 3
    assert result["cpu_gpu_scope_attribution"]["mode"] == "gpu_only_fallback"
    assert (
        result["forward_window_inference"]["fallback_role"]
        == "primary_gpu_only_heuristic"
    )
    assert all(result["measurement_invariants"].values())


def test_hidden_comm_excludes_same_stream_and_includes_other_stream(tmp_path):
    trace_path = tmp_path / "stream-aware-TP-0.trace.json.gz"
    write_gzip_trace(
        trace_path,
        [
            gpu_event("ep_dispatch_intranode_0", 0, 10, 1),
            # Timestamp overlap on the comm stream is serialization/artifact,
            # not hidden communication.
            gpu_event("same_stream_compute", 0, 10, 1),
            # Only this three-microsecond interval can hide communication.
            gpu_event("other_stream_compute", 5, 3, 2),
        ],
    )

    communication = ANALYZER.analyze_rank_trace(trace_path)["communication"]

    assert communication["union_us"] == 10
    assert communication["hidden_by_other_stream_non_comm_us"] == 3
    assert communication["exposed_us"] == 7
    dispatch = communication["phases"]["flydsl_dispatch"]
    assert dispatch["hidden_by_other_stream_non_comm_us"] == 3
    assert dispatch["exposed_us"] == 7


def test_duration_inflation_matches_only_same_normalized_kernel_name():
    events = [
        gpu_event("Matched   Kernel", 0, 10, 0),
        gpu_event("matched kernel", 20, 12, 0),
        gpu_event("MATCHED KERNEL", 100, 20, 0),
        gpu_event("matched kernel", 140, 24, 0),
        gpu_event("overlap_helper", 100, 20, 1),
        gpu_event("overlap_helper", 140, 24, 1),
        gpu_event("unmatched_short", 200, 1, 0),
    ]

    inflation = ANALYZER._duration_inflation(events, minimum_samples_per_state=2)

    assert inflation["matched_kernel_count"] == 1
    matched = inflation["kernels"][0]
    assert matched["canonical_name"] == "matched kernel"
    assert matched["overlapped_count"] == 2
    assert matched["unoverlapped_count"] == 2
    assert matched["p50_ratio"] == 2
    assert inflation["aggregate"]["median_p50_ratio"] == 2


def test_forward_confidence_uses_layer_marker_multiplicity():
    markers = [(0, 1), (10, 11), (100_000, 100_001), (100_010, 100_011)]
    inference = ANALYZER.infer_forward_windows(markers, (0, 100_020), num_layers=2)

    assert inference["confidence"] == "medium"
    assert inference["layer_validation"]["marker_counts_per_window"] == [2, 2]
    assert inference["layer_validation"]["plausible_equal_marker_counts"]
    assert not inference["layer_validation"]["cpu_forward_span_evidence"]


def test_cpu_scope_mapping_uses_correlation_and_ac2g_flow():
    events = [
        {
            "name": "a0",
            "cat": "user_annotation",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 0,
            "dur": 100,
            "args": {"External id": 1},
        },
        {
            "name": "dispatch_a",
            "cat": "user_annotation",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 10,
            "dur": 40,
            "args": {"External id": 2},
        },
        {
            "name": "hipLaunchKernel",
            "cat": "cuda_runtime",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 20,
            "dur": 1,
            "args": {"External id": 3, "correlation": 7},
        },
        {
            "name": "ac2g",
            "cat": "ac2g",
            "ph": "s",
            "id": 7,
            "pid": 100,
            "tid": 100,
            "ts": 20,
            "args": {},
        },
        {
            "name": "ac2g",
            "cat": "ac2g",
            "ph": "f",
            "id": 7,
            "pid": 8,
            "tid": 1,
            "ts": 60,
            "args": {},
        },
        {
            "name": "hipStreamWaitEvent",
            "cat": "cuda_runtime",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 25,
            "dur": 2,
            "args": {"correlation": 8},
        },
    ]
    gpu_events = [gpu_event("ep_dispatch_intranode_0", 60, 5, 1)]
    gpu_events[0]["args"]["correlation"] = 7
    gpu_events[0]["args"]["External id"] = 3

    analysis = ANALYZER.analyze_cpu_gpu_scopes(events, gpu_events)

    assert analysis["mode"] == "cpu_gpu_scope_mapping"
    assert analysis["coverage"]["stage_mapping_ratio"] == 1
    assert analysis["coverage"]["operation_mapping_ratio"] == 1
    assert analysis["coverage"]["ac2g_flow_link_ratio"] == 1
    assert analysis["coverage"]["confidence"] == "high"
    stage = analysis["stage_scopes"][0]
    operation = analysis["operation_scopes"][0]
    assert stage["mapping_methods"] == {"correlation_ac2g": 1}
    assert operation["communication"]["union_us"] == 5
    assert operation["event_wait_latency"]["count"] == 1
    assert operation["event_wait_latency"]["dependency_latency_us"] is None


def test_cpu_scope_mapping_external_id_fallback_reports_medium_confidence():
    events = [
        {
            "name": "b0",
            "cat": "user_annotation",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 0,
            "dur": 100,
            "args": {"External id": 1},
        },
        {
            "name": "aten::empty",
            "cat": "cpu_op",
            "ph": "X",
            "pid": 100,
            "tid": 100,
            "ts": 20,
            "dur": 1,
            "args": {"External id": 99},
        },
    ]
    gpu_events = [gpu_event("compute", 60, 5, 1)]
    gpu_events[0]["args"]["External id"] = 99

    analysis = ANALYZER.analyze_cpu_gpu_scopes(events, gpu_events)

    assert analysis["coverage"]["stage_mapping_ratio"] == 1
    assert analysis["coverage"]["ac2g_flow_link_ratio"] == 0
    assert analysis["coverage"]["confidence"] == "medium"
    assert analysis["stage_scopes"][0]["mapping_methods"] == {"external_id": 1}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ep_dispatch_intranode_0", "flydsl_dispatch"),
        ("flydsl_combine_kernel", "flydsl_combine"),
        ("ncclDevKernel_Generic_1", "dp_nccl"),
        ("flashattnfwdcombine", "non_comm"),
        ("ordinary_dispatch", "non_comm"),
    ],
)
def test_phase_classification(name, expected):
    assert ANALYZER.classify_phase(name, "kernel") == expected


def test_rank_aggregation_reports_requested_percentiles():
    ranks = [
        {
            "rank": rank,
            "gpu_wall_span_us": value,
            "communication": {"union_us": value * 2},
        }
        for rank, value in enumerate((10, 20, 30, 40))
    ]

    aggregate = ANALYZER.aggregate_ranks(ranks)

    assert aggregate["gpu_wall_span_us"] == {
        "rank_count": 4,
        "min": 10.0,
        "p50": 25.0,
        "median": 25.0,
        "p95": pytest.approx(38.5),
        "max": 40.0,
    }
    assert aggregate["communication.union_us"]["p50"] == 50
