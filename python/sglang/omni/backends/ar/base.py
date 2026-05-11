# SPDX-License-Identifier: Apache-2.0
"""Fallback autoregressive backend used when no AR runtime is configured."""

from sglang.omni.protocol import (
    ARBackend,
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniRequest,
)
from sglang.omni.streaming import OmniStreamSink


class UnsupportedARBackend(ARBackend):
    """AR backend placeholder that fails at request time."""

    def begin_request_context(
        self,
        request: OmniRequest,
        *,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle:
        raise RuntimeError("No omni AR backend is configured")

    def append_input_segments(
        self,
        context: OmniContextBundle,
        request: OmniRequest,
        *,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle:
        raise RuntimeError("No omni AR backend is configured")

    def decode_until_boundary(
        self,
        context: OmniContextBundle,
        *,
        request: OmniRequest,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniBoundary:
        raise RuntimeError("No omni AR backend is configured")

    def append_generated_segment(
        self,
        context: OmniContextBundle,
        segment: GeneratedSegment,
        *,
        request: OmniRequest,
    ) -> OmniContextBundle:
        raise RuntimeError("No omni AR backend is configured")

    def get_context_ops(self, context: OmniContextBundle):
        raise RuntimeError("No omni AR backend is configured")

    def release(self, context: OmniContextBundle) -> None:
        return None
