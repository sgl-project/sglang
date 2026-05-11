# SPDX-License-Identifier: Apache-2.0
"""Executor-backed multimodal_gen backend for future omni diffusion runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.omni.backends.mm_gen.pipeline_forward_backend import (
    build_pipeline_req,
    coerce_generated_segment,
)
from sglang.omni.protocol import (
    ContextOps,
    GeneratedSegment,
    MultimodalGenerationBackend,
    OmniRequest,
)


@dataclass(slots=True)
class PipelineExecutorBackend(MultimodalGenerationBackend):
    """A backend that drives multimodal_gen stages through a multimodal_gen.PipelineExecutor.

    This is the boundary to use when omni stops calling a full pipeline object
    directly and lets a diffusion-serving runtime own stage scheduling,
    parallelism, and component residency.
    """

    executor: PipelineExecutor
    stages: Sequence[Any]
    server_args: Any

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment:
        batch = build_pipeline_req(request, context_ops)
        batch = self.executor.execute_with_profiling(
            list(self.stages),
            batch,
            self.server_args,
        )
        segment = batch.generated_segment
        if segment is None:
            raise ValueError("Pipeline executor did not set generated_segment")
        return coerce_generated_segment(segment)
