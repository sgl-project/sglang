"""Per-commit streaming-session tests.

Default config + EagleV2RetractLargePage + abort-leak repro stay per-commit.
Other variants (Retract / Eagle / EagleV2 / EagleRetractLargePage) live in
test_streaming_session_extra.py.
"""

import unittest

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
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=691, stage="stage-b", runner_config="1-gpu-large")


class TestStreamingSession(StreamingSessionServerBase, StreamingSessionKitMixin):
    """Default streaming-session config (small model, no spec)."""

    extra_args = ["--chunked-prefill-size", "512"]


class TestStreamingSessionEagleV2RetractLargePage(TestStreamingSession):
    """EAGLE3 spec v2 + retract + page=256."""

    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
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
    ]
    env_overrides = [
        ("SGLANG_ENABLE_SPEC_V2", True),
        ("SGLANG_TEST_RETRACT", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


class TestStreamingSessionAbortLeakRepro(
    StreamingSessionServerBase, AbortLeakReproKitMixin
):
    extra_args = [
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
    ]


if __name__ == "__main__":
    unittest.main()
