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

register_cuda_ci(est_time=519, stage="stage-b", runner_config="1-gpu-large")


SWA_MODEL = "openai/gpt-oss-20b"

# Common gpt-oss-20b launch args. Matches TestSessionLatency/TestSWARadixCacheKL.
SWA_COMMON_ARGS = [
    "--mem-fraction-static",
    "0.70",
    "--disable-piecewise-cuda-graph",
]


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


class TestStreamingSessionSWARetractMixedChunk(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """SWA under retract decode with --enable-mixed-chunk."""

    model = SWA_MODEL
    extra_args = [
        "--chunked-prefill-size",
        "128",
        "--enable-mixed-chunk",
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
