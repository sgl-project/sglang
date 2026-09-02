"""Label-gated vision/omni server launches too expensive for the per-commit
budget; the per-commit set lives in test_vision_openai_server_a.py."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import OmniOpenAITestMixin

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")


class TestQwen3OmniServer(OmniOpenAITestMixin):
    model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    extra_args = [  # workaround to fit into H100
        "--mem-fraction-static=0.90",
        "--disable-cuda-graph",
        "--image-processor-backend=pil",
        "--grammar-backend=none",
    ]


# Delete the mixin so it is not collected as a test case in its own right.
del OmniOpenAITestMixin


if __name__ == "__main__":
    unittest.main()
