# SPDX-License-Identifier: Apache-2.0
"""Executor-backed multimodal_gen backend for future omni diffusion runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sglang.omni.mm_gen.pipeline_forward import (
    build_pipeline_req,
    coerce_generated_segment,
)
from sglang.omni.protocol import ContextOps, GeneratedSegment, OmniRequest


@dataclass(slots=True)
class PipelineExecutorBackend:
    """Drive multimodal_gen stages through a PipelineExecutor.

    This is the boundary to use when omni stops calling a full pipeline object
    directly and lets a diffusion-serving runtime own stage scheduling,
    parallelism, and component residency.
    """

    executor: Any
    stages: Sequence[Any]
    server_args: Any
    context_ops_extra_key: str

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment:
        batch = build_pipeline_req(request, context_ops, self.context_ops_extra_key)
        batch = self.executor.execute_with_profiling(
            list(self.stages),
            batch,
            self.server_args,
        )
        segment = batch.generated_segment
        if segment is None:
            raise ValueError("Pipeline executor did not set generated_segment")
        return coerce_generated_segment(segment)
