import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_script(
    "diffusion_nightly_runner",
    "scripts/ci/utils/diffusion/run_comparison.py",
)
dashboard = _load_script(
    "diffusion_nightly_dashboard",
    "scripts/ci/utils/diffusion/generate_diffusion_dashboard.py",
)


def test_sglang_server_warmup_matches_measured_shape():
    case = {
        "model": "example/model",
        "num_gpus": 2,
        "width": 768,
        "height": 512,
        "num_frames": 121,
    }
    command = runner._build_sglang_cmd(
        case,
        {"serve_args": "--warmup-mode server --tp-size 2"},
        30000,
    )

    resolution_index = command.index("--warmup-resolutions")
    frame_index = command.index("--warmup-num-frames")
    assert command[resolution_index + 1] == "768x512"
    assert command[frame_index + 1] == "121"


def test_explicit_server_warmup_shape_is_preserved():
    case = {
        "model": "example/model",
        "num_gpus": 1,
        "width": 1024,
        "height": 1024,
        "num_frames": 81,
    }
    command = runner._build_sglang_cmd(
        case,
        {
            "serve_args": (
                "--warmup-mode server --warmup-resolutions 512x512 "
                "--warmup-num-frames 25"
            )
        },
        30000,
    )

    assert command.count("--warmup-resolutions") == 1
    assert command.count("--warmup-num-frames") == 1
    assert command[command.index("--warmup-resolutions") + 1] == "512x512"
    assert command[command.index("--warmup-num-frames") + 1] == "25"


def test_perf_dump_summary_uses_medians():
    perf_dumps = [
        {
            "total_duration_ms": 1000.0,
            "steps": [
                {"name": "TextEncodingStage", "duration_ms": 100.0},
                {"name": "DenoisingStage", "duration_ms": 800.0},
            ],
            "denoise_steps_ms": [{"duration_ms": 8.0}, {"duration_ms": 10.0}],
        },
        {
            "total_duration_ms": 3000.0,
            "steps": [
                {"name": "TextEncodingStage", "duration_ms": 300.0},
                {"name": "DenoisingStage", "duration_ms": 2400.0},
            ],
            "denoise_steps_ms": [{"duration_ms": 30.0}],
        },
        {
            "total_duration_ms": 1100.0,
            "steps": [
                {"name": "TextEncodingStage", "duration_ms": 110.0},
                {"name": "DenoisingStage", "duration_ms": 880.0},
            ],
            "denoise_steps_ms": [{"duration_ms": 11.0}],
        },
    ]

    summary = runner._summarize_perf_dumps(perf_dumps)

    assert summary["server_latency_s"] == 1.1
    assert summary["server_stage_medians_ms"] == {
        "DenoisingStage": 880.0,
        "TextEncodingStage": 110.0,
    }
    assert summary["median_denoise_step_ms"] == 10.5


def test_dashboard_uses_historical_median_and_shows_server_breakdown():
    current = {
        "timestamp": "2026-09-04T00:00:00+00:00",
        "commit_sha": "abcdef123456",
        "results": [
            {
                "case_id": "example",
                "framework": "sglang",
                "model": "example/model",
                "latency_s": 10.4,
                "latency_samples_s": [10.3, 10.4, 10.5],
                "measurement_count": 3,
                "server_latency_s": 10.0,
                "server_stage_medians_ms": {
                    "TextEncodingStage": 100.0,
                    "DenoisingStage": 9800.0,
                    "DecodingStage": 100.0,
                },
                "median_denoise_step_ms": 196.0,
            }
        ],
    }
    history = [
        {
            "results": [
                {
                    "case_id": "example",
                    "framework": "sglang",
                    "latency_s": value,
                }
            ]
        }
        for value in (10.0, 30.0, 9.8)
    ]

    baseline, count = dashboard._historical_latency_baseline(
        "example", "sglang", history
    )
    markdown, alerts = dashboard.generate_dashboard(current, history)

    assert baseline == 10.0
    assert count == 3
    assert alerts == []
    assert "| 3 | **10.40** |" in markdown
    assert "## SGLang Server-Side Breakdown" in markdown
    assert "| model | 10.00 | 0.10 | 9.80 | 0.10 | 196.00 |" in markdown
