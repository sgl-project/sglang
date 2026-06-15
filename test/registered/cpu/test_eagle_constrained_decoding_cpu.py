import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cpu_ci(est_time=900, suite="base-b-test-cpu")


class TestEagleConstrainedDecoding(
    CustomTestCase, RegexConstrainedMixin, JSONConstrainedMixin
):
    max_running_requests = 64
    attention_backend = "intel_amx"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6
    page_size = 1
    other_launch_args = []
    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    grammar_backend = "xgrammar"
    # Run the synchronous (non-overlap) scheduling path. This is the only
    # path on CPU: overlap scheduling is force-disabled for speculative
    # decoding on CPU, so the overlap-scheduling variant of the CUDA suite
    # (TestEagleConstrainedDecodingV2) is not mirrored here.
    disable_overlap = True

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # popen_launch_server auto-appends ``--device cpu`` on a CPU-only host,
        # and the CPU CI environment provides SGLANG_USE_CPU_ENGINE=1, so
        # neither needs to be set here (same as the sibling CPU spec tests).
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
            # On CPU the static memory pool is sized from host RAM per NUMA
            # node, so use the smaller fraction shared by the CPU CI suite.
            "--mem-fraction-static",
            "0.3",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--grammar-backend",
            cls.grammar_backend,
        ]
        if cls.disable_overlap:
            launch_args.append("--disable-overlap-schedule")
        launch_args.extend(cls.other_launch_args)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
