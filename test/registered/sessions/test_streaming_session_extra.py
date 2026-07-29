import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=691, stage="extra-a", runner_config="1-gpu-large")


class TestStreamingSessionRetractMixedChunk(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """Retract + --enable-mixed-chunk."""

    extra_args = ["--chunked-prefill-size", "128", "--enable-mixed-chunk"]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestStreamingSessionRetractLargePage(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """Retract + page=256: exercises page-aligned `_free_tail`. Partial-page
    free would corrupt pages still holding committed tokens."""

    extra_args = ["--chunked-prefill-size", "4096", "--page-size", "256"]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


# Common EAGLE3 spec args; reused by Eagle/EagleV2/EagleRetractLargePage variants.
_EAGLE3_SPEC_ARGS = [
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
]


class TestStreamingSessionEagle(StreamingSessionServerBase, StreamingSessionKitMixin):
    """EAGLE3 spec v1 (overlap disabled); offset=-1 — see kit's note."""

    kv_inherit_offset = -1
    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "512",
        *_EAGLE3_SPEC_ARGS,
    ]
    env_overrides = [("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True)]


class TestStreamingSessionEagleV2(StreamingSessionServerBase, StreamingSessionKitMixin):
    """EAGLE3 spec v2 (overlap on)."""

    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
        "--chunked-prefill-size",
        "512",
        *_EAGLE3_SPEC_ARGS,
    ]
    env_overrides = [
        ("SGLANG_ENABLE_SPEC_V2", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


class TestStreamingSessionEagleRetractLargePage(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """EAGLE3 spec v1 + retract + page=256: max-pressure on `_free_tail`
    (spec tail + retract alloc-commit gap + page alignment)."""

    kv_inherit_offset = -1
    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "4096",
        *_EAGLE3_SPEC_ARGS,
        "--page-size",
        "256",
    ]
    env_overrides = [
        ("SGLANG_TEST_RETRACT", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


if __name__ == "__main__":
    unittest.main()
