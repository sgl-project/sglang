import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_TARGET_MODEL_STANDALONE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class TestStandaloneConstrainedDecoding(
    CustomTestCase, RegexConstrainedMixin, JSONConstrainedMixin
):
    max_running_requests = 64
    attention_backend = "triton"
    spec_steps = 4
    spec_topk = 1
    spec_draft_tokens = 5
    page_size = 1
    model = DEFAULT_TARGET_MODEL_STANDALONE
    draft_model = DEFAULT_DRAFT_MODEL_STANDALONE
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
            "STANDALONE",
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
            "0.7",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--grammar-backend",
            cls.grammar_backend,
        ]
        if cls.disable_overlap:
            launch_args.append("--disable-overlap-schedule")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestStandaloneConstrainedDecodingOverlap(TestStandaloneConstrainedDecoding):
    # Overlap scheduling: grammar decode goes through the grammar barrier.
    disable_overlap = False


if __name__ == "__main__":
    unittest.main()
