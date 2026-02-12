import unittest
import os

from sglang.srt.utils import kill_process_tree
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.test_utils import DEFAULT_DRAFT_MODEL_EAGLE, DEFAULT_TARGET_MODEL_EAGLE

ACC_THRESHOLDS = {
    DEFAULT_TARGET_MODEL_EAGLE: {"kl_div": 0.0001},
}


class TestSpecV2(CustomTestCase):
    spec_num_steps = 3
    spec_eagle_topk = 1
    spec_num_draft_tokens = 4
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        str(spec_num_steps),
        "--speculative-eagle-topk",
        str(spec_eagle_topk),
        "--speculative-num-draft-tokens",
        str(spec_num_draft_tokens),
        "--disable-radix-cache",
    ]
    env = {
        **os.environ,
        "SGLANG_ENABLE_SPEC_V2": "1",
    }
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_output_logprobs_match(self):
        test_input_output_logprobs_match_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=512,
            max_prompt_tokens=2000,
        )

class TestSpecV1Ref(TestSpecV2):
    env = None
    

if __name__ == "__main__":
    unittest.main()
