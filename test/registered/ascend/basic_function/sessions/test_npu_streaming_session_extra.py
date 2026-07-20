"""
Streaming session extra tests for NPU.

Ported from test/registered/sessions/test_streaming_session_extra.py.
Adapted for Ascend NPU backend:
  - Replaces CUDA model paths with NPU-local weights. Uses Qwen3-8B +
    Qwen3-8B_eagle3 for EAGLE3 variants (Llama-3.1-8B + EAGLE3-Llama-3.1-8B
    is incompatible with NPU aclnnRmsNorm: the draft's RMSNorm weight is
    float16 while the main model hidden state is bfloat16, a combination
    not in the supported list).
  - Adds `--attention-backend ascend`, `--disable-cuda-graph`,
    `--disable-piecewise-cuda-graph` to all server launches.
  - Sets `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` and a longer
    `HCCL_EXEC_TIMEOUT` for stability under multi-turn streaming workloads.
  - Adapts `--page-size` to NPU-friendly values (128/4 instead of 256),
    since Ascend's page_size default of 128 must divide `chunked_prefill_size`.

Tests:
  - TestNPUStreamingSessionRetractMixedChunk: retract + mixed-chunk
  - TestNPUStreamingSessionRetractLargePage: retract + page=128
  - TestNPUStreamingSessionEagle: EAGLE3 spec v1 (overlap disabled)
  - TestNPUStreamingSessionEagleV2: EAGLE3 spec v2 (overlap on, default env)
  - TestNPUStreamingSessionEagleRetractLargePage: EAGLE3 + retract + page=128
"""

import os
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=900, suite="full-1-npu-a3", nightly=True)


class NPUStreamingSessionExtraServerBase(StreamingSessionServerBase):
    """Base server fixture for NPU streaming-session extra tests.

    Mirrors `NPUStreamingSessionServerBase` in test_npu_streaming_session.py
    so the two files stay consistent. Lives locally to keep the file
    self-contained.
    """

    npu_env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_EXEC_TIMEOUT": "200",
    }

    @classmethod
    def setUpClass(cls):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1)
            )
            stack.enter_context(envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(True))
            for name, val in cls.env_overrides:
                stack.enter_context(getattr(envs, name).override(val))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-streaming-session"] + list(cls.extra_args),
                env=cls.npu_env,
            )
        cls.tokenizer = get_tokenizer(cls.model)


# NPU-common server args shared by all variants below. Kept as a list so
# subclasses can extend (not replace) them via unpacking.
_NPU_COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-piecewise-cuda-graph",
    "--enable-streaming-session",
    "--mem-fraction-static",
    "0.7",
]


class TestNPUStreamingSessionRetractMixedChunk(
    NPUStreamingSessionExtraServerBase, StreamingSessionKitMixin
):
    """Retract + --enable-mixed-chunk.

    `chunked-prefill-size` must be divisible by NPU page_size (128) to
    avoid the scheduler assertion; 128 itself satisfies this and keeps
    prefill chunks small enough to exercise mixed-chunk interleaving.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        *_NPU_COMMON_ARGS,
        "--chunked-prefill-size",
        "128",
        "--enable-mixed-chunk",
        "--page-size",
        "128",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestNPUStreamingSessionRetractLargePage(
    NPUStreamingSessionExtraServerBase, StreamingSessionKitMixin
):
    """Retract + page=128: exercises page-aligned `_free_tail`.

    Partial-page free would corrupt pages still holding committed tokens.
    The CUDA original uses page=256; on Ascend we use the NPU default of
    128 (256 is not supported by the ascend backend). `chunked-prefill-size`
    4096 is divisible by 128.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        *_NPU_COMMON_ARGS,
        "--chunked-prefill-size",
        "4096",
        "--page-size",
        "128",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


# Common EAGLE3 spec args; reused by Eagle/EagleV2/EagleRetractLargePage variants.
# Uses NPU-local Qwen3-8B_eagle3 draft weights (Llama-3.1-8B + EAGLE3-Llama-3.1-8B
# is incompatible with NPU aclnnRmsNorm — see module docstring).
_EAGLE3_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


class TestNPUStreamingSessionEagle(
    NPUStreamingSessionExtraServerBase, StreamingSessionKitMixin
):
    """EAGLE3 spec v1 (overlap disabled); offset=-1 — see kit's note."""

    kv_inherit_offset = -1
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        *_NPU_COMMON_ARGS,
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "512",
        *_EAGLE3_SPEC_ARGS,
        "--page-size",
        "4",
    ]
    env_overrides = [("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True)]


class TestNPUStreamingSessionEagleV2(
    NPUStreamingSessionExtraServerBase, StreamingSessionKitMixin
):
    """EAGLE3 spec v2 (overlap on).

    Note: `SGLANG_ENABLE_SPEC_V2` defaults to True in current sglang; no
    explicit env override needed. Older CI images may not have this attr
    on `Envs`, so overriding it would raise `AttributeError`.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        *_NPU_COMMON_ARGS,
        "--chunked-prefill-size",
        "512",
        *_EAGLE3_SPEC_ARGS,
        "--page-size",
        "4",
    ]
    env_overrides = [("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True)]


class TestNPUStreamingSessionEagleRetractLargePage(
    NPUStreamingSessionExtraServerBase, StreamingSessionKitMixin
):
    """EAGLE3 spec v1 + retract + page=128: max-pressure on `_free_tail`
    (spec tail + retract alloc-commit gap + page alignment)."""

    kv_inherit_offset = -1
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        *_NPU_COMMON_ARGS,
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "4096",
        *_EAGLE3_SPEC_ARGS,
        "--page-size",
        "128",
    ]
    env_overrides = [
        ("SGLANG_TEST_RETRACT", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


__all__ = [
    "TestNPUStreamingSessionRetractMixedChunk",
    "TestNPUStreamingSessionRetractLargePage",
    "TestNPUStreamingSessionEagle",
    "TestNPUStreamingSessionEagleV2",
    "TestNPUStreamingSessionEagleRetractLargePage",
]


if __name__ == "__main__":
    unittest.main()
