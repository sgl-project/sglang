"""Nightly coverage of the GDN speculative ReplaySSM ring under unified memory,
where the ring and its cursors live in the arena's request sub-pool."""

import unittest

from test_qwen35_fp4_mtp import MTP_BASE_ARGS, QWEN35_FP4_MODEL, _run_mtp_gsm8k

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.reasoning_kit import ReasoningTokenUsageMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=400, stage="nightly", runner_config="4-gpu-b200")


class TestQwen35FP4MTPReplaySSMUnifiedMemory(ReasoningTokenUsageMixin, CustomTestCase):
    """MTP with the ReplaySSM spec ring under --enable-unified-memory."""

    reasoning_parser_name = "qwen3"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_FP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.init_reasoning_token_verifier()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=MTP_BASE_ARGS
            + [
                "--enable-linear-replayssm-spec",
                "--enable-unified-memory",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_mtp_gsm8k(self)


if __name__ == "__main__":
    unittest.main()
