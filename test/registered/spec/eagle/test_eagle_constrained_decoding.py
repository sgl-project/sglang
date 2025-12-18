import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.json_constrained_kit import TestJSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import TestRegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=100, suite="stage-b-test-small-1-gpu")


class TestEagleConstrainedDecoding(
    CustomTestCase, TestRegexConstrainedMixin, TestJSONConstrainedMixin
):
    max_running_requests = 64
    attention_backend = "triton"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6
    page_size = 1
    other_launch_args = []
    model = DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST
    draft_model = DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST
    grammar_backend = "xgrammar"
    eagle_v2 = False

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model",
            cls.draft_model,
            "--speculative-num-steps",
            cls.spec_steps,
            "--speculative-eagle-topk",
            cls.spec_topk,
            "--speculative-num-draft-tokens",
            cls.spec_draft_tokens,
            "--page-size",
            str(cls.page_size),
            "--mem-fraction-static",
            "0.75",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--grammar-backend",
            cls.grammar_backend,
        ]
        launch_args.extend(cls.other_launch_args)
        with envs.SGLANG_ENABLE_SPEC_V2.override(cls.eagle_v2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestEagleConstrainedDecodingV2(TestEagleConstrainedDecoding):
    eagle_v2 = True


if __name__ == "__main__":
    unittest.main()
