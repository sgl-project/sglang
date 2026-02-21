"""
Usage:
python3 -m unittest test_vision_openai_server_tp2.TestQwen25VLTP2Server.test_single_image_chat_completion
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import ImageOpenAITestMixin, VideoOpenAITestMixin

# Register for 2-GPU per-commit CI
register_cuda_ci(est_time=300, suite="stage-b-test-large-2-gpu")


class TestQwen25VLTP2Server(ImageOpenAITestMixin, VideoOpenAITestMixin):
    """Test Qwen2.5-VL-7B with TP=2."""

    model = "Qwen/Qwen2.5-VL-7B-Instruct"
    extra_args = [
        "--cuda-graph-max-bs=4",
        "--tp=2",
    ]


class TestInternVL25TP2Server(ImageOpenAITestMixin):
    """Test InternVL2.5-8B with TP=2."""

    model = "OpenGVLab/InternVL2_5-8B"
    extra_args = [
        "--cuda-graph-max-bs=4",
        "--tp=2",
    ]


# Delete the mixin classes so that they are not collected by pytest
del (
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
