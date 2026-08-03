from summarize import build_report, latency_summary, summarize_runs


def _run(concurrency, p95, fps, error_rate=0.0):
    return {
        "concurrency": concurrency,
        "action_to_first_frame_ms": {
            "p50": p95 * 0.8,
            "p95": p95,
            "p99": p95 * 1.1,
        },
        "chunk_total_ms": {"p95": p95},
        "aggregate_fps": fps,
        "min_session_fps": fps,
        "error_rate": error_rate,
    }


def test_report_selects_highest_concurrency_that_meets_slo():
    report = summarize_runs(
        [_run(1, p95=700, fps=18), _run(2, p95=920, fps=16.5), _run(4, p95=1400, fps=12)]
    )
    assert report["max_supported_concurrency"] == 2


def test_report_rejects_high_aggregate_fps_when_one_session_is_slow():
    run = _run(4, p95=800, fps=64)
    run["min_session_fps"] = 12
    report = summarize_runs([run])
    assert report["max_supported_concurrency"] == 0


def test_report_calculates_async_improvement_at_common_concurrency():
    report = build_report(
        {"runs": [_run(1, p95=800, fps=16)]},
        {"runs": [_run(1, p95=600, fps=18)]},
    )
    assert report["comparison"]["async_improvement_pct"] == 25.0


def test_latency_summary_uses_nearest_rank_percentiles():
    summary = latency_summary(range(1, 101))
    assert summary["p50"] == 50.5
    assert summary["p95"] == 95
    assert summary["p99"] == 99
