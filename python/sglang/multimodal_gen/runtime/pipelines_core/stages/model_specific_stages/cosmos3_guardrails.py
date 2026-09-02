# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 guardrail stages.

Text and video safety checks via the ``cosmos_guardrail`` package.
Install with: pip install cosmos-guardrail==0.3.1

Enabled by default when available; opt out with
``SGLANG_DISABLE_COSMOS3_GUARDRAILS=1``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_checker = None


@lru_cache(maxsize=1)
def is_cosmos_guardrail_available() -> bool:
    return importlib.util.find_spec("cosmos_guardrail") is not None


def _mirror_symlinked_nltk_data() -> None:
    """Make the guardrail's nltk_data readable under NLTK's hardened opener.

    ``CosmosSafetyChecker`` registers its HF-hub snapshot's
    ``blocklist/nltk_data`` directory on ``nltk.data.path``. Hub snapshot files
    are symlinks into the blob store, which NLTK builds that ship the
    ``pathsec`` hardened opener refuse to follow (O_NOFOLLOW, CWE-59 TOCTOU
    guard) — every text-safety check then fails with "refusing to follow a
    symlink at open time". Mirror each symlink-containing search entry to a
    plain-file copy and register the mirror ahead of the original, so NLTK's
    data lookup resolves to real files first. No-op for NLTK builds without
    the hardened opener and for search entries that are already plain files.
    """
    try:
        import nltk.data
    except ImportError:
        return

    mirror_root = Path(envs.SGLANG_DIFFUSION_CACHE_ROOT) / "nltk_data_deref"
    for entry in list(nltk.data.path):
        try:
            root = Path(entry)
            if not root.is_dir():
                continue
            if not any(p.is_symlink() for p in root.rglob("*")):
                continue
            mirror = mirror_root / hashlib.sha256(str(root).encode()).hexdigest()[:16]
            if str(mirror) in nltk.data.path:
                continue
            if not mirror.is_dir():
                # Every GPU worker process runs this at pipeline construction on
                # a shared filesystem, so stage under a per-process name and
                # publish with an atomic rename; whichever process publishes
                # first wins and the others adopt its mirror.
                staging = mirror.with_name(f"{mirror.name}.{os.getpid()}.tmp")
                shutil.rmtree(staging, ignore_errors=True)
                # symlinks=False dereferences: the copy holds real file contents.
                shutil.copytree(root, staging, symlinks=False)
                try:
                    staging.rename(mirror)
                except OSError:
                    shutil.rmtree(staging, ignore_errors=True)
                    if not mirror.is_dir():
                        raise
            nltk.data.path.insert(nltk.data.path.index(entry), str(mirror))
            logger.info(
                "Mirrored symlinked nltk_data %s -> %s (hardened-NLTK compatibility)",
                root,
                mirror,
            )
        except OSError as exc:
            # Best-effort: an unwritable cache root must not break guardrail
            # init. Hardened-NLTK builds may still fail at check time; plain
            # NLTK builds work fine without the mirror.
            logger.warning(
                "Could not mirror symlinked nltk_data %s under %s: %s",
                entry,
                mirror_root,
                exc,
            )


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
    _mirror_symlinked_nltk_data()
    idle_device = "cpu" if offload_to_cpu else current_platform.device_type
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
