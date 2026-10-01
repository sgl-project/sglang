import importlib.util
import io
from pathlib import Path

from PIL import Image

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


def test_sglang_launch_is_exactly_the_recipe():
    # An outside benchmark launches the published recipe as-is, so the harness
    # must not add shape-specific warmup flags of its own.
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

    assert "--warmup-resolutions" not in command
    assert "--warmup-num-frames" not in command
    assert command[-4:] == ["--warmup-mode", "server", "--tp-size", "2"]


def test_requests_of_a_run_never_repeat_an_input(monkeypatch):
    # sglang reuses the encodings of inputs it has seen, so a repeated prompt
    # or image would time that reuse instead of the encoders
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 704), (200, 100, 50)).save(buffer, format="PNG")
    monkeypatch.setattr(runner, "_cached_ref_image", buffer.getvalue())
    monkeypatch.setattr(runner, "_ref_image_variants", {})
    case = {"prompt": "Make the cat wear a red hat"}

    requests = [runner._request_case(case, index) for index in range(10)]
    prompts = [request["prompt"] for request in requests]
    downscaled = [
        Image.open(
            io.BytesIO(runner._get_ref_image_variant({}, request["input_variant"]))
        )
        .resize((256, 176), Image.BICUBIC)
        .tobytes()
        for request in requests
    ]

    assert len(set(prompts)) == len(set(downscaled)) == 10
    assert len({len(prompt) for prompt in prompts}) == 1
    assert case == {"prompt": "Make the cat wear a red hat"}


class _FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_sglang_video_latency_includes_polling_and_download(monkeypatch):
    statuses = iter(["queued", "in_progress", "completed"])
    fetched = []

    class FakeRequests:
        @staticmethod
        def post(url, json=None, timeout=None):
            return _FakeResponse({"id": "job-1"})

        @staticmethod
        def get(url, timeout=None):
            fetched.append(url)
            if url.endswith("/content"):
                return _FakeResponse(content=b"mp4-bytes")
            return _FakeResponse({"status": next(statuses)})

    sleeps = []
    monkeypatch.setattr(runner, "requests", FakeRequests)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    case = {"model": "example/model", "prompt": "p", "width": 64, "height": 64}

    latency = runner.send_video_request_sglang("http://host", case)

    assert latency >= 0
    assert fetched[-1] == "http://host/v1/videos/job-1/content"
    assert fetched.count("http://host/v1/videos/job-1") == 3
    assert sleeps == [runner.POLL_INTERVAL_S] * 2


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


def test_health_wait_fails_fast_when_the_server_exits(monkeypatch):
    class ExitedProc:
        returncode = 1

        def poll(self):
            return self.returncode

    def unreachable(*args, **kwargs):
        raise runner.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(runner.requests, "get", unreachable)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    try:
        runner.wait_for_health("http://host:1", proc=ExitedProc(), timeout=3600)
    except RuntimeError as e:
        assert "exited with code 1" in str(e)
    else:
        raise AssertionError("wait_for_health should fail once the server exits")


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
        "methodology": "client-e2e-v1",
        "warmup_requests": 1,
        "results": [
            {
                "case_id": "example",
                "framework": "sglang",
                "model": "example/model",
                "first_request_latency_s": 25.0,
                "latency_s": 10.4,
                "latency_samples_s": [10.3, 10.4, 10.5],
                "measurement_count": 3,
                "server_latency_samples_s": [9.9, 10.0],
                "server_latency_s": 10.0,
                "missing_perf_dumps": 1,
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
    assert "Incomplete Server Telemetry" in markdown
    assert "**model**: 2/3 server samples available" in markdown
    assert "| 3 | 2/3 | 25.00 | **10.40** |" in markdown
    assert "Methodology `client-e2e-v1`" in markdown
    assert "## SGLang Server-Side Breakdown" in markdown
    assert "| model | 10.00 | 0.10 | 9.80 | 0.10 | 196.00 |" in markdown
