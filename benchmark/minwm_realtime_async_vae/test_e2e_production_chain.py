import pytest

from e2e_production_chain import (
    REQUIRED_TRACE_EVENTS,
    ProductionGateError,
    render_chinese_report,
    validate_production_result,
)


def _result():
    return {
        "hardware": {
            "denoiser": {"instance_type": "p5.4xlarge", "gpu": "H100", "count": 1},
            "vae": {"instance_type": "g6.2xlarge", "gpu": "L4", "count": 1},
        },
        "runs": [
            {
                "concurrency": concurrency,
                "successful_sessions": concurrency,
                "errors": [],
                "error_rate": 0.0,
                "trace_event_names": sorted(REQUIRED_TRACE_EVENTS),
                "direct_vae_frame_batches": 8,
                "chunk_total_ms": {"p50": 480.0, "p95": 520.0},
                "action_to_first_frame_ms": {"p50": 520.0, "p95": 620.0},
                "stage_ms": {
                    "denoise_ms": {"p50": 420.0, "p95": 450.0},
                    "vae_decode_ms": {"p50": 18.0, "p95": 22.0},
                },
            }
            for concurrency in (1, 4)
        ],
        "browser": {"display_lag_ms": {"p50": 92.0, "p95": 184.0}},
    }


def test_production_result_requires_all_roles_direct_media_and_display_budget():
    result = _result()

    validate_production_result(result)

    result["runs"][1]["trace_event_names"].remove("coordinator.admit_complete")
    with pytest.raises(ProductionGateError, match="coordinator.admit_complete"):
        validate_production_result(result)

    result = _result()
    result["runs"][0]["direct_vae_frame_batches"] = 0
    with pytest.raises(ProductionGateError, match="direct VAE"):
        validate_production_result(result)

    result = _result()
    result["browser"]["display_lag_ms"]["p95"] = 251.0
    with pytest.raises(ProductionGateError, match="display lag"):
        validate_production_result(result)


def test_chinese_report_contains_hardware_concurrency_and_stage_timings():
    report = render_chinese_report(_result(), run_id="run-1")

    assert "H100" in report
    assert "L4" in report
    assert "4 并发" in report
    assert "Denoise" in report
    assert "VAE Decode" in report
    assert "Display Lag" in report
