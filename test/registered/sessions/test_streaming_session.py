"""Per-commit streaming-session tests.

Default config + EagleV2RetractLargePage + abort-leak repro stay per-commit.
Other variants (Retract / Eagle / EagleV2 / EagleRetractLargePage) live in
test_streaming_session_extra.py.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.streaming_session_kit import (
    AbortLeakReproKitMixin,
    StreamingSessionKitMixin,
)
from sglang.test.server_fixtures.streaming_session_fixture import (
    ABORT_REPRO_CHUNKED_PREFILL_SIZE,
    ABORT_REPRO_CONTEXT_LEN,
    ABORT_REPRO_PAGE_SIZE,
    StreamingSessionServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=691, stage="stage-b", runner_config="1-gpu-large")


class TestStreamingSession(StreamingSessionServerBase, StreamingSessionKitMixin):
    """Default streaming-session config (small model, no spec)."""


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


class TestStreamingSessionAbortLeakRepro(StreamingSessionServerBase, AbortLeakReproKitMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
                    "--context-length",
                    str(ABORT_REPRO_CONTEXT_LEN),
                    "--page-size",
                    str(ABORT_REPRO_PAGE_SIZE),
                    "--max-running-requests",
                    "32",
                    "--log-level",
                    "info",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
