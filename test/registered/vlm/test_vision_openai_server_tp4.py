"""
Usage:
python3 -m unittest test_vision_openai_server_tp4.TestInternVL25TP4Server.test_single_image_chat_completion
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import ImageOpenAITestMixin

# Nightly multi-GPU coverage (4 GPUs)
register_cuda_ci(est_time=400, suite="nightly-4-gpu", nightly=True)


class TestInternVL25TP4Server(ImageOpenAITestMixin):
    """Test InternVL2.5-8B with TP=4."""

    model = "OpenGVLab/InternVL2_5-8B"
    extra_args = [
        "--cuda-graph-max-bs=4",
        "--tp=4",
    ]


# Delete the mixin classes so that they are not collected by pytest
del (ImageOpenAITestMixin,)


if __name__ == "__main__":
    unittest.main()

