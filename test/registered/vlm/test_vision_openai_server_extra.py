"""Label-gated vision/omni server launches too expensive for the per-commit
budget; the per-commit set lives in test_vision_openai_server_a.py."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.vlm_utils import OmniOpenAITestMixin, TestOpenAIMLLMServerBase

register_cuda_ci(est_time=239, stage="extra-a", runner_config="1-gpu-large")


class TestQwen3OmniServer(OmniOpenAITestMixin):
    # Exact cache-hit output equality needs batch-invariant operators. Keep the
    # remaining modality tests on production defaults and check reuse below.
    test_image_prefix_cache_reuse = None
    model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    extra_args = [  # workaround to fit into H100
        "--mem-fraction-static=0.90",
        "--disable-cuda-graph",
        "--image-processor-backend=pil",
        "--grammar-backend=none",
    ]


class TestQwen3OmniPrefixCacheServer(TestOpenAIMLLMServerBase):
    model = TestQwen3OmniServer.model
    extra_args = [
        *TestQwen3OmniServer.extra_args,
        "--enable-deterministic-inference",
        # Deterministic FlashInfer attention disables radix caching. Triton
        # keeps prefix reuse enabled on both Hopper and Blackwell.
        "--attention-backend=triton",
    ]
    test_image_prefix_cache_reuse = OmniOpenAITestMixin.test_image_prefix_cache_reuse


# Avoid collecting the imported base classes as independent test cases.
del OmniOpenAITestMixin, TestOpenAIMLLMServerBase


if __name__ == "__main__":
    unittest.main()
