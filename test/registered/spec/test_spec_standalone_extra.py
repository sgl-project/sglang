import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.standalone_fixture import StandaloneServerBase
from sglang.test.test_utils import CustomTestCase

# Non-V2 standalone speculative decoding tests (FA3, Triton, FlashInfer
# backends). Sibling V2 classes stay per-commit in test_spec_standalone.py.
register_cuda_ci(est_time=406, stage="extra-a", runner_config="1-gpu-large")


class TestStandaloneSpeculativeDecodingBase(StandaloneServerBase, CustomTestCase):
    attention_backend = "fa3"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    enable_spec_v2 = False


class TestStandaloneSpeculativeDecodingTriton(StandaloneServerBase, CustomTestCase):
    attention_backend = "triton"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    enable_spec_v2 = False


class TestStandaloneSpeculativeDecodingFlashinfer(StandaloneServerBase, CustomTestCase):
    attention_backend = "flashinfer"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    enable_spec_v2 = False


if __name__ == "__main__":
    unittest.main()
