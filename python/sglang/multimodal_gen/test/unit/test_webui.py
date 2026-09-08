# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.apps.webui.minimax_h3 import (
    _generate_minimax_h3,
    build_minimax_h3_sampling_params_kwargs,
    minimax_h3_tasks_for_server,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)


def _h3_kwargs(**overrides):
    values = {
        "prompt": "A quiet city street with synchronized ambient audio",
        "task": "t2va",
        "first_frame": None,
        "last_frame": None,
        "reference_image": None,
        "reference_video": None,
        "reference_audio": None,
        "seed": 42,
        "num_inference_steps": 50,
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    values.update(overrides)
    return build_minimax_h3_sampling_params_kwargs(**values)


def test_h3_webui_tasks_follow_loaded_partition():
    assert minimax_h3_tasks_for_server(
        SimpleNamespace(model_variant="fl2va", model_subfolder=None)
    ) == ("t2va", "fl2va")
    assert minimax_h3_tasks_for_server(
        SimpleNamespace(model_variant="ref2va", model_subfolder=None)
    ) == ("ref2va",)


def test_h3_t2va_uses_native_contract_without_generic_cfg_fields():
    kwargs = _h3_kwargs()

    assert kwargs["conditions"] == []
    assert kwargs["target"] == {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
    }
    assert {
        "negative_prompt",
        "guidance_scale",
        "num_frames",
        "fps",
        "width",
        "height",
        "enable_teacache",
    }.isdisjoint(kwargs)
    assert minimax_h3_validate_canonical_request(**kwargs)["task"] == "t2va"


def test_h3_fl2va_maps_first_and_last_keyframes_in_order(tmp_path):
    first_frame = tmp_path / "first.png"
    kwargs = _h3_kwargs(
        task="fl2va",
        first_frame=first_frame,
        last_frame="https://example.com/last.png",
        aspect_ratio="auto",
    )

    assert kwargs["conditions"] == [
        {
            "type": "image",
            "uri": first_frame.resolve().as_uri(),
            "role": "keyframe",
            "frame_index": 0,
        },
        {
            "type": "image",
            "uri": "https://example.com/last.png",
            "role": "keyframe",
            "frame_index": -1,
        },
    ]
    canonical = minimax_h3_validate_canonical_request(**kwargs)
    assert [item["frame_index"] for item in canonical["conditions"]] == [0, -1]


def test_h3_ref2va_maps_multimodal_references_in_order():
    kwargs = _h3_kwargs(
        task="ref2va",
        reference_image="https://example.com/person.png",
        reference_video="https://example.com/motion.mp4",
        reference_audio="https://example.com/voice.wav",
    )

    assert [(item["type"], item["role"]) for item in kwargs["conditions"]] == [
        ("image", "reference"),
        ("video_audio", "reference"),
        ("audio", "reference"),
    ]
    assert minimax_h3_validate_canonical_request(**kwargs)["task"] == "ref2va"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "fl2va"}, "requires a first frame"),
        ({"task": "ref2va"}, "requires a reference"),
        (
            {"task": "t2va", "reference_image": "https://example.com/a.png"},
            "does not accept conditioning media",
        ),
    ],
)
def test_h3_webui_rejects_invalid_task_media(overrides, message):
    with pytest.raises(ValueError, match=message):
        _h3_kwargs(**overrides)


def _video_request():
    return SimpleNamespace(
        data_type=DataType.VIDEO,
        fps=24,
        save_output=True,
        output_compression=50,
        output_file_path=Mock(return_value="/tmp/generated.mp4"),
    )


def test_h3_generation_runs_video_lifecycle_and_preserves_audio():
    server_args = SimpleNamespace(model_path="MiniMaxAI/MiniMax-H3")
    sampling_params = Mock()
    request = _video_request()
    result = SimpleNamespace(
        output=[object()],
        audio=object(),
        audio_sample_rate=24000,
        error=None,
    )

    with (
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3."
            "SamplingParams.from_user_sampling_params_args",
            return_value=sampling_params,
        ),
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3.prepare_request",
            return_value=request,
        ),
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3.sync_scheduler_client.forward",
            return_value=result,
        ),
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3.save_outputs",
            return_value=["/tmp/generated.mp4"],
        ) as save_outputs,
    ):
        output = _generate_minimax_h3(server_args, {"prompt": "test"})

    assert output == "/tmp/generated.mp4"
    sampling_params.prepare_video_request_for_queue.assert_called_once_with(request)
    sampling_params.validate_video_final_outputs.assert_called_once_with(
        ["/tmp/generated.mp4"], request
    )
    sampling_params.cleanup_video_request.assert_called_once_with(request)
    assert save_outputs.call_args.kwargs["audio"] is result.audio
    assert save_outputs.call_args.kwargs["audio_sample_rate"] == 24000


def test_h3_generation_cleans_up_after_scheduler_error():
    server_args = SimpleNamespace(model_path="MiniMaxAI/MiniMax-H3")
    sampling_params = Mock()
    request = _video_request()

    with (
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3."
            "SamplingParams.from_user_sampling_params_args",
            return_value=sampling_params,
        ),
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3.prepare_request",
            return_value=request,
        ),
        patch(
            "sglang.multimodal_gen.apps.webui.minimax_h3.sync_scheduler_client.forward",
            return_value=SimpleNamespace(error="generation failed"),
        ),
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        _generate_minimax_h3(server_args, {"prompt": "test"})

    sampling_params.cleanup_video_request.assert_called_once_with(request)
