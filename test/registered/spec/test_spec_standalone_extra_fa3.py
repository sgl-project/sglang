"""FA3 class of test_spec_standalone_extra.py, kept on H100 because FA3 is SM80/SM90 only."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.standalone_fixture import StandaloneServerBase
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=80, stage="extra-a", runner_config="1-gpu-large")


class TestStandaloneSpeculativeDecodingBase(StandaloneServerBase, CustomTestCase):
    attention_backend = "fa3"
    speculative_eagle_topk = 2
    speculative_num_draft_tokens = 7
    disable_overlap = True


if __name__ == "__main__":
    unittest.main()
