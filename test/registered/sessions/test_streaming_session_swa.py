"""Per-commit streaming-session tests on a hybrid-SWA model.

Baseline + large-page retract + abort-leak repro stay per-commit; the
mixed-chunk retract variant lives in test_streaming_session_swa_extra.py.
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
    SWA_COMMON_ARGS,
    SWA_MODEL,
    StreamingSessionServerBase,
)

register_cuda_ci(est_time=332, stage="base-b", runner_config="1-gpu-large")


class TestStreamingSessionSWA(StreamingSessionServerBase, StreamingSessionKitMixin):
    """Baseline streaming session on a hybrid-SWA model."""

    model = SWA_MODEL
    extra_args = ["--chunked-prefill-size", "512", *SWA_COMMON_ARGS]


class TestStreamingSessionSWARetractLargePage(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """SWA under retract decode with page=256."""

    model = SWA_MODEL
    extra_args = [
        "--chunked-prefill-size",
        "4096",
        "--page-size",
        "256",
        *SWA_COMMON_ARGS,
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestStreamingSessionSWAAbortLeakRepro(
    StreamingSessionServerBase, AbortLeakReproKitMixin
):
    """SWA abort-heavy chunked prefill leak repro."""

    model = SWA_MODEL
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
        *SWA_COMMON_ARGS,
    ]


if __name__ == "__main__":
    unittest.main()
