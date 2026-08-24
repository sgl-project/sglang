"""Label-gated SWA streaming-session variants.

CUDA-only: gpt-oss-20b is not part of the AMD streaming-session coverage.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin
from sglang.test.server_fixtures.streaming_session_fixture import (
    SWA_COMMON_ARGS,
    SWA_MODEL,
    StreamingSessionServerBase,
)

register_cuda_ci(est_time=147, stage="extra-a", runner_config="1-gpu-large")


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


if __name__ == "__main__":
    unittest.main()
