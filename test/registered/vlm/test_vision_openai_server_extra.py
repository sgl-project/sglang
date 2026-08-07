"""Label-gated OpenAI-API vision/omni server launches.

Holds the launches that are too expensive for the per-commit 1-gpu-h100 budget;
the per-commit set lives in test_vision_openai_server_a.py.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import *
from sglang.test.vlm_utils import (
    AudioOpenAITestMixin,
    ImageOpenAITestMixin,
    OmniOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    VideoOpenAITestMixin,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")


class TestQwen3OmniServer(OmniOpenAITestMixin):
    model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    extra_args = [  # workaround to fit into H100
        "--mem-fraction-static=0.90",
        "--disable-cuda-graph",
        "--disable-fast-image-processor",
        "--grammar-backend=none",
    ]


# Delete the mixin classes so that they are not collected by pytest
del (
    TestOpenAIMLLMServerBase,
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
    AudioOpenAITestMixin,
    OmniOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
