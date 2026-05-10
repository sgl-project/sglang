# SPDX-License-Identifier: Apache-2.0
"""Multimodal generation backends used by omni orchestration."""

from sglang.omni.mm_gen.pipeline_executor import PipelineExecutorBackend
from sglang.omni.mm_gen.pipeline_forward import DirectPipelineForwardBackend

__all__ = [
    "DirectPipelineForwardBackend",
    "PipelineExecutorBackend",
]
