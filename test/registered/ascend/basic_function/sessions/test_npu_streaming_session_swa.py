"""Streaming session tests on a hybrid-SWA model (gpt-oss-20b) on NPU.

Ported from test/sessions/test_streaming_session_swa.py.

NPU adaptations:
- model: openai/gpt-oss-20b -> openai-mirror/gpt-oss-20b (ModelScope mirror)
- --cuda-graph-backend-prefill=disabled -> --disable-cuda-graph
- --page-size 256 -> 128 (NPU page_size constraint)
- added --attention-backend ascend
"""

import unittest

from sglang.test.ascend.test_ascend_utils import GPT_OSS_120B_BF16_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import (
    AbortLeakReproKitMixin,
    StreamingSessionKitMixin,
)
from sglang.test.server_fixtures.streaming_session_fixture import (
    ABORT_REPRO_CHUNKED_PREFILL_SIZE,
    ABORT_REPRO_CONTEXT_LEN,
    StreamingSessionServerBase,
)

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


SWA_MODEL = GPT_OSS_120B_BF16_WEIGHTS_PATH

# NPU adaptation: use bf16 gpt-oss-120b instead of mxfp4 gpt-oss-20b to avoid
# quantization compatibility issues. Parameters referenced from
# test_npu_gpt_oss_120b_bf16.py.
SWA_COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.7",
    "--attention-backend",
    "ascend",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--max-running-requests",
    "32",
    "--watchdog-timeout",
    "9000",
    "--tp-size",
    "8",
    "--sampling-backend",
    "ascend",
    "--disable-cuda-graph",
]


class TestStreamingSessionSWA(StreamingSessionServerBase, StreamingSessionKitMixin):
    """Baseline streaming session on a hybrid-SWA model.

    [Test Category] Memory_and_Scheduling
    [Test Target] --enable-streaming-session;hybrid-SWA
    """

    model = SWA_MODEL
    extra_args = ["--chunked-prefill-size", "512", *SWA_COMMON_ARGS]


class TestStreamingSessionSWARetractLargePage(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """SWA under retract decode with page=128 (NPU constraint; CUDA used 256).

    [Test Category] Memory_and_Scheduling
    [Test Target] --enable-streaming-session;SGLANG_TEST_RETRACT;--page-size
    """

    model = SWA_MODEL
    extra_args = [
        "--chunked-prefill-size",
        "4096",
        "--page-size",
        "128",
        *SWA_COMMON_ARGS,
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestStreamingSessionSWARetractMixedChunk(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """SWA under retract decode with --enable-mixed-chunk.

    [Test Category] Memory_and_Scheduling
    [Test Target] --enable-streaming-session;SGLANG_TEST_RETRACT;--enable-mixed-chunk
    """

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
    """SWA abort-heavy chunked prefill leak repro.

    [Test Category] Memory_and_Scheduling
    [Test Target] --enable-streaming-session;abort leak
    """

    model = SWA_MODEL
    # NPU adaptation: page-size 256 -> 128
    extra_args = [
        "--chunked-prefill-size",
        str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
        "--context-length",
        str(ABORT_REPRO_CONTEXT_LEN),
        "--page-size",
        "128",
        "--max-running-requests",
        "32",
        "--log-level",
        "info",
        *SWA_COMMON_ARGS,
    ]


if __name__ == "__main__":
    unittest.main()
