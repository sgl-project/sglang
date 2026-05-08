# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.omni.protocol import (
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniRequest,
)


class UnsupportedARBackend:
    def prepare_context(self, request: OmniRequest) -> OmniContextBundle:
        raise RuntimeError("No omni AR backend is configured")

    def decode_until_boundary(
        self,
        context: OmniContextBundle,
        *,
        request: OmniRequest,
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


class UnsupportedGenerationBackend:
    def generate_segment(self, request: OmniRequest, context_ops) -> GeneratedSegment:
        raise RuntimeError("No omni generation backend is configured")
