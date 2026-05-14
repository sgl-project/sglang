"""Per-commit streaming-session tests.

Only TestStreamingSessionEagleV2RetractLargePage and (the imported)
TestStreamingSessionAbortLeakRepro stay per-commit. Everything else
(default base + Retract* + Eagle*/EagleV2 + EagleRetractLargePage) is
registered through test_streaming_session_extra.py.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.streaming_session_fixture import (
    TestStreamingSession,
    TestStreamingSessionAbortLeakRepro,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=691, stage="stage-b", runner_config="1-gpu-large")


class TestStreamingSessionEagleV2RetractLargePage(TestStreamingSession):
    """EAGLE3 spec v2 + retract + page=256."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
            2
        ), envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "4096",
                    "--dtype=float16",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model",
                    DEFAULT_DRAFT_MODEL_EAGLE3,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                    "--page-size",
                    "256",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
