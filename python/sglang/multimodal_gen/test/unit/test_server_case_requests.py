from types import SimpleNamespace

from sglang.multimodal_gen.test.server import test_server_utils
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
)


class _FakeVideos:
    def __init__(self):
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id="request-id")

    def list(self):
        return SimpleNamespace(
            data=[SimpleNamespace(id="request-id", status="completed")]
        )

    def download_content(self, *, video_id):
        assert video_id == "request-id"
        return SimpleNamespace(read=lambda: b"video")


def test_direct_url_video_request_preserves_workload_extras(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(test_server_utils, "validate_openai_video", lambda _data: None)
    monkeypatch.setattr(
        test_server_utils,
        "validate_video_file",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        test_server_utils,
        "upload_file_to_slack",
        lambda **_kwargs: None,
    )
    videos = _FakeVideos()
    client = SimpleNamespace(videos=videos)
    sampling_params = DiffusionSamplingParams(
        prompt="prompt",
        image_path="https://example.com/input.png",
        direct_url_test=True,
        output_size="384x640",
        num_frames=17,
        fps=16,
        extras={
            "num_inference_steps": 12,
            "guidance_scale": 4.5,
            "seed": 0,
        },
    )

    generate = test_server_utils.get_generate_fn(
        model_path="test/model",
        modality="video",
        sampling_params=sampling_params,
    )
    request_id, content = generate("case", client)

    assert request_id == "request-id"
    assert content == b"video"
    assert videos.create_kwargs["extra_body"] == {
        "reference_url": "https://example.com/input.png",
        "fps": 16,
        "num_frames": 17,
        "num_inference_steps": 12,
        "guidance_scale": 4.5,
        "seed": 0,
    }
