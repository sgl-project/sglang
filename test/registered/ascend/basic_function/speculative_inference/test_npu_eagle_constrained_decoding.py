import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestEagleConstrainedDecoding(
    CustomTestCase, RegexConstrainedMixin, JSONConstrainedMixin
):
    """Testcase: EAGLE3 spec-decoding constrained decoding test (non-overlap).
    Validates correctness of speculative decoding with grammar-based constraints
    under non-overlapped scheduling. Covers regex and JSON constrained generation.

    [Test Category] Functionality
    [Test Target] EAGLE3 spec-decoding with xgrammar backend (non-overlap schedule)
    """

    max_running_requests = 8
    attention_backend = "ascend"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6
    page_size = 128
    other_launch_args = []
    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    grammar_backend = "xgrammar"
    # Run the synchronous (non-overlap) scheduling path.
    disable_overlap = True

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
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
            "0.85",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--grammar-backend",
            cls.grammar_backend,
            "--chunked-prefill-size",
            1024,
            "--dtype",
            "bfloat16",
        ]
        if cls.disable_overlap:
            launch_args.append("--disable-overlap-schedule")
        launch_args.extend(cls.other_launch_args)
        env = {
            **os.environ,
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestEagleConstrainedDecodingV2(TestEagleConstrainedDecoding):
    """Testcase: EAGLE3 spec-decoding constrained decoding test (overlap).
    Validates correctness of speculative decoding with grammar-based constraints
    under overlapped scheduling (Spec v2). Covers regex and JSON constrained generation.

    [Test Category] Functionality
    [Test Target] EAGLE3 spec-decoding with xgrammar backend (overlap schedule)
    """

    disable_overlap = False


if __name__ == "__main__":
    unittest.main()
