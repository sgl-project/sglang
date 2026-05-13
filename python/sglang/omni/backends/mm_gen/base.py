# SPDX-License-Identifier: Apache-2.0
"""Fallback multimodal generation backend used by unsupported omni servers."""

from sglang.omni.core.protocol import (
    ContextOps,
    GeneratedSegment,
    MultimodalGenerationBackend,
    OmniRequest,
)


class UnsupportedGenerationBackend(MultimodalGenerationBackend):
    """Generation backend placeholder that fails at request time."""

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment:
        raise RuntimeError("No omni generation backend is configured")
