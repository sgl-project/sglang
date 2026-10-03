from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionSamplingParams


def test_url_video_request_preserves_sampling_extras():
    extras = {
        "profile": True,
        "num_profiled_timesteps": 5,
        "num_inference_steps": 12,
        "seed": 0,
    }
    original_extras = extras.copy()
    params = DiffusionSamplingParams(
        prompt="test",
        image_path="https://example.com/input.png",
        direct_url_test=True,
        fps=24,
        num_frames=25,
        extras=extras,
    )
    client = Mock()
    client.videos.create.side_effect = ConnectionError("stop at transport boundary")
    generate = get_generate_fn("test-model", "video", params)
    with pytest.raises(ConnectionError, match="stop at transport boundary"):
        generate("url-video", client)
    assert client.videos.create.call_args.kwargs["extra_body"] == {
        "reference_url": params.image_path,
        "fps": 24,
        "num_frames": 25,
        **original_extras,
    }
    assert params.extras == original_extras
