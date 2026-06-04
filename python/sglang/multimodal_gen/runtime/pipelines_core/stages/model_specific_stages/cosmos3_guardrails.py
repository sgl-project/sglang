# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 guardrail stages.

Text and video safety checks via the ``cosmos_guardrail`` package.
Install with: pip install cosmos-guardrail==0.3.1

Enabled by default when available; opt out with
``SGLANG_DISABLE_COSMOS3_GUARDRAILS=1``.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache

import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_checker = None


@lru_cache(maxsize=1)
def is_cosmos_guardrail_available() -> bool:
    return importlib.util.find_spec("cosmos_guardrail") is not None


def _init_guardrails(offload_to_cpu: bool = False) -> None:
    global _checker
    if _checker is not None:
        return
    try:
        from cosmos_guardrail import CosmosSafetyChecker
    except ImportError:
        raise ImportError(
            "cosmos_guardrail is required for Cosmos3 safety checks. "
            "Install it with: pip install cosmos-guardrail==0.3.1"
        )
    logger.info(
        "Initializing Cosmos3 guardrails (offload_to_cpu=%s) ...", offload_to_cpu
    )
    _checker = CosmosSafetyChecker()
    idle_device = "cpu" if offload_to_cpu else "cuda"
    for runner in (_checker.text_guardrail, _checker.video_guardrail):
        if runner is None or not hasattr(runner, "models"):
            continue
        for m in runner.models:
            if isinstance(m, torch.nn.Module):
                m.to(idle_device)
    logger.info("Cosmos3 guardrails initialized.")


def check_text_safety(prompt: str) -> None:
    if _checker is None:
        return
    if not _checker.check_text_safety(prompt):
        raise ValueError("Guardrail blocked prompt.")


def check_video_safety(video: np.ndarray) -> np.ndarray:
    """Apply video guardrails to decoded frames.

    Args:
        video: numpy [B, T, H, W, C] or [T, H, W, C], uint8.

    Returns:
        Processed frames in the same shape, or raises ValueError if blocked.
    """
    if _checker is None:
        return video
    if video.ndim == 5:
        processed = []
        for frames in video:
            result = _checker.check_video_safety(frames)
            processed.append(result if result is not None else frames)
        return np.stack(processed)
    result = _checker.check_video_safety(video)
    return result if result is not None else video


class Cosmos3TextGuardrailStage(PipelineStage):
    """Check prompt text against safety policies before generation.

    Raises ``ValueError`` if the prompt is blocked.
    """

    parallelism_type = StageParallelismType.MAIN_RANK_ONLY

    def __init__(self, offload_to_cpu: bool = False):
        super().__init__()
        _init_guardrails(offload_to_cpu)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.use_guardrails is False:
            return batch
        prompt = batch.prompt
        if prompt is None:
            return batch
        if isinstance(prompt, list):
            for p in prompt:
                check_text_safety(p)
        else:
            check_text_safety(prompt)
        return batch
