import hashlib
import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.benchmarks.bench_offline_throughput import (
    BatchOutput,
    BenchArgs,
    RequestOutput,
    calculate_metrics,
    generate_batch,
)
from sglang.multimodal_gen.benchmarks.request_manifest import load_request_manifest


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
        BenchArgs(),
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
