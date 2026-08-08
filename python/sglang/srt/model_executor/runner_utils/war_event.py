"""WAR read-done event utilities for CUDA graph runners."""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadBoundary
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


def make_war_read_done_event(device_module) -> Optional[torch.cuda.Event]:
    """Create a persistent external event for CUDA graph capture."""
    if not is_cuda():
        return None
    try:
        return device_module.Event(external=True)
    except TypeError:
        return None


def maybe_publish_prefill_war_read_done(
    model_runner, forward_batch, device_module
) -> None:
    """Publish prefill read-done after compliant metadata initialization."""
    if not envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.get():
        return
    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return
    # TODO(Jialin): Relax this gate for speculative decoding after its prefill
    # WAR boundaries are validated.
    if not model_runner.spec_algorithm.is_none():
        return
    # The record lands right after replay prep, so PRE_REPLAY only.
    boundary = model_runner.attn_backend.shared_read_boundary(
        forward_batch.forward_mode
    )
    if boundary is not SharedReadBoundary.PRE_REPLAY:
        return
    logger.info_once(
        "Prefill WAR read-done fastpath active (%s)",
        type(model_runner.attn_backend).__name__,
    )
    read_done = device_module.Event()
    read_done.record()
    model_runner.war_fastpath_read_done_event = read_done
