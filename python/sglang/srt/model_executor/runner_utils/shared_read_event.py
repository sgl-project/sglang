"""Shared-read-done event utilities for graph and eager runners."""

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


def make_external_event(device_module) -> Optional[torch.cuda.Event]:
    """Create a persistent external event, e.g., for CUDA graph capture."""
    if not is_cuda():
        return None
    try:
        return device_module.Event(external=True)
    except TypeError:
        return None


def maybe_publish_prefill_shared_read_done(
    model_runner, shared_read_ends: SharedReadEnds, device_module
) -> None:
    """Publish prefill read-done at the resolved pre-replay boundary."""
    if shared_read_ends is not SharedReadEnds.PRE_REPLAY:
        return
    logger.info_once(
        "Prefill shared-read-done fastpath active (%s)",
        type(model_runner.attn_backend).__name__,
    )
    read_done = device_module.Event()
    read_done.record()
    model_runner.shared_read_done_event = read_done
