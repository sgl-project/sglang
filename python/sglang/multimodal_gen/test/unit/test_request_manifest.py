import argparse
import asyncio
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.benchmarks import bench_offline_throughput
from sglang.multimodal_gen.benchmarks.bench_offline_throughput import (
    BatchOutput,
    BenchArgs,
    RequestOutput,
    calculate_metrics,
    generate_batch,
    throughput_test,
)
from sglang.multimodal_gen.benchmarks.bench_serving import (
    async_request_image_sglang,
    async_request_video_sglang,
)
from sglang.multimodal_gen.benchmarks.datasets import RequestFuncInput
from sglang.multimodal_gen.benchmarks.request_manifest import load_request_manifest
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_load_request_manifest_resolves_inputs_and_preserves_overrides(tmp_path):
    image_path = tmp_path / "condition.png"
    image_path.write_bytes(b"condition-image")
    manifest_path = tmp_path / "workload.jsonl"
    records = [
        {
            "request_id": "i2v-1",
            "prompt": "Make the water move",
            "image_paths": "condition.png",
            "seed": 7,
            "num_frames": 17,
            "sampling_params": {"flow_shift": 5.0},
        },
        {
            "prompt": "A lighthouse in a storm",
            "image_paths": [
                "https://example.com/reference.png",
                "condition.png",
            ],
            "width": 832,
            "height": 480,
        },
    ]
    _write_jsonl(manifest_path, records)

    manifest = load_request_manifest(manifest_path)

    assert manifest.path == str(manifest_path.resolve())
    assert manifest.sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert [request.request_id for request in manifest.requests] == [
        "i2v-1",
        "request-00002",
    ]
    assert manifest.requests[0].sampling_params == {
        "flow_shift": 5.0,
        "seed": 7,
        "num_frames": 17,
        "image_path": str(image_path.resolve()),
    }
    assert manifest.requests[1].sampling_params["image_path"] == [
        "https://example.com/reference.png",
        str(image_path.resolve()),
    ]


@pytest.mark.parametrize(
    ("record", "error"),
    [
        ({"prompt": ""}, "prompt must be a non-empty string"),
        ({"prompt": "test", "typo": 1}, "unsupported field"),
        (
            {"prompt": "test", "sampling_params": {"output_path": "/tmp"}},
            "cannot set reserved",
        ),
        (
            {"prompt": "test", "image_paths": []},
            "image_paths must be a non-empty",
        ),
    ],
)
def test_load_request_manifest_rejects_invalid_records(tmp_path, record, error):
    manifest_path = tmp_path / "invalid.jsonl"
    _write_jsonl(manifest_path, [record])

    with pytest.raises(ValueError, match=error):
        load_request_manifest(manifest_path)


def test_load_request_manifest_rejects_duplicate_ids(tmp_path):
    manifest_path = tmp_path / "duplicate.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {"request_id": "same", "prompt": "first"},
            {"request_id": "same", "prompt": "second"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate request_id"):
        load_request_manifest(manifest_path)


def test_load_request_manifest_rejects_empty_file(tmp_path):
    manifest_path = tmp_path / "empty.jsonl"
    manifest_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no requests"):
        load_request_manifest(manifest_path)


def test_metrics_use_the_successful_request_shapes_after_a_failure():
    failed = RequestOutput(
        request_id="failed",
        prompt="first",
        sampling_params={"width": 1024, "height": 1024, "num_frames": 81},
        success=False,
        latency_seconds=1.0,
        error="expected failure",
    )
    succeeded = RequestOutput(
        request_id="succeeded",
        prompt="second",
        sampling_params={"width": 832, "height": 480, "num_frames": 17},
        success=True,
        latency_seconds=2.0,
    )
    batch = BatchOutput(
        num_samples=1,
        total_frames=17,
        success=True,
        requests=[failed, succeeded],
    )

    metrics = calculate_metrics(
        [batch],
        total_duration=3.0,
        resolution=(32, 32, 1),
        num_requests=2,
        all_sampling_params=[failed.sampling_params, succeeded.sampling_params],
    )

    assert metrics["successful_requests"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["total_pixels_generated"] == 832 * 480 * 17


def test_generate_batch_records_request_id_output_path_and_digest(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "generated.mp4"
    output_path.write_bytes(b"generated-video")

    class FakeDeviceModule:
        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 0

    class FakeEngine:
        @staticmethod
        def generate(sampling_params_kwargs):
            assert sampling_params_kwargs["request_id"] == "video-1"
            return SimpleNamespace(output_file_path=output_path)

    monkeypatch.setattr(
        "sglang.multimodal_gen.benchmarks.bench_offline_throughput.torch.get_device_module",
        lambda: FakeDeviceModule,
    )

    output = generate_batch(
        FakeEngine(),
        prompts=["A moving test pattern"],
        user_sampling_params=[{"num_frames": 17, "seed": 42}],
        request_ids=["video-1"],
    )

    assert output.success
    assert output.num_samples == 1
    assert output.total_frames == 17
    assert output.requests[0].request_id == "video-1"
    assert output.requests[0].output_file_paths == [str(output_path)]
    assert output.requests[0].output_sha256 == [
        hashlib.sha256(output_path.read_bytes()).hexdigest()
    ]


def test_offline_benchmark_runs_with_disable_safety_checker(monkeypatch):
    """--disable-safety-checker must not add a request field SamplingParams
    rejects: every request then fails and the run reports zero successes."""

    class FakeDeviceModule:
        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 0

    class FakeEngine:
        server_args = SimpleNamespace(pipeline_config=SimpleNamespace(task_type="t2i"))

        @staticmethod
        def generate(sampling_params_kwargs):
            SamplingParams(**sampling_params_kwargs)
            return SimpleNamespace(output_file_path=None)

    monkeypatch.setattr(
        bench_offline_throughput.torch, "get_device_module", lambda: FakeDeviceModule
    )
    monkeypatch.setattr(
        bench_offline_throughput, "initialize_engine", lambda server_args: FakeEngine()
    )
    monkeypatch.setattr(
        bench_offline_throughput, "configure_logger", lambda server_args: None
    )
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--disable-safety-checker", "--num-prompts", "1", "--disable-tqdm"]
    )

    metrics = throughput_test(
        SimpleNamespace(model_path="m"), BenchArgs.from_cli_args(args)
    )

    assert metrics["failed_requests"] == 0
    assert metrics["successful_requests"] == 1


class _RejectedSubmit:
    status = 503

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def text(self):
        return "rejected"


class _SubmitRecorder:
    """Stands in for the HTTP session: keeps the submitted form and rejects it."""

    def __init__(self):
        self.form = None

    def post(self, url, data=None, json=None):
        self.form = data
        return _RejectedSubmit()


@pytest.mark.parametrize(
    "request_func", [async_request_image_sglang, async_request_video_sglang]
)
def test_bench_serving_upload_requests_send_num_inference_steps(tmp_path, request_func):
    """Requests that upload an image go out as multipart forms; a form without the
    step count runs the model default while the SLO model assumes the requested one."""
    image_path = tmp_path / "condition.png"
    image_path.write_bytes(b"condition-image")
    session = _SubmitRecorder()

    asyncio.run(
        request_func(
            RequestFuncInput(
                prompt="Make the water move",
                api_url="http://127.0.0.1:1/v1/unused",
                model="m",
                image_paths=[str(image_path)],
                num_inference_steps=3,
            ),
            session,
        )
    )

    fields = {options["name"]: value for options, _, value in session.form._fields}
    for value in fields.values():
        if isinstance(value, io.IOBase):
            value.close()
    assert fields["num_inference_steps"] == "3"
