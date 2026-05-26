import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.hybrid_attn_backend_fixture import (
    TestHybridAttnBackendBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
)

# Hybrid attention backend tests (FA3 prefill + FlashInfer decode, requires SM 90+ / H100)
# Multiple test classes: base, MLA, TorchCompile, SpecDecode variants
register_cuda_ci(est_time=407, stage="extra-a", runner_config="1-gpu-large")


class TestHybridAttnBackendMLA(TestHybridAttnBackendBase):
    accuracy_threshold = 0.60
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA


class TestHybridAttnBackendTorchCompile(TestHybridAttnBackendBase):
    accuracy_threshold = 0.65
    extra_args = ["--enable-torch-compile"]


class TestHybridAttnBackendSpeculativeDecodingPrefillBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2
    extra_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "2",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-attention-mode",
        "prefill",
    ]


class TestHybridAttnBackendSpeculativeDecodingDecodeBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2
    extra_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "2",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-attention-mode",
        "decode",
    ]


if __name__ == "__main__":
    unittest.main()
