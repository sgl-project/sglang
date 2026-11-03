"""Shared-read-done event utilities for graph and eager runners."""

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import is_cuda, is_npu

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUSharedReadEvent,
    )

logger = logging.getLogger(__name__)


def make_external_event(
    device_module,
) -> Optional[Union[torch.cuda.Event, "NPUSharedReadEvent"]]:
    """Create a persistent external event, e.g., for CUDA graph capture."""
    if is_npu():
        from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
            NPUSharedReadEvent,
        )

        return NPUSharedReadEvent(device_module)
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
    if not model_runner.spec_algorithm.supports_prefill_shared_read_done():
        return
    # Host and storage offload copy a stashed chunk's KV ordered only behind the
    # schedule stream, so they need the whole-forward barrier unless every such
    # copy waits for the forward stream itself.
    memory = get_memory()
    if memory.enable_lmcache or (
        (memory.enable_hierarchical_cache or memory.hicache_storage_backend is not None)
        and not model_runner.kv_offload_fenced_on_forward_stream
    ):
        return
    logger.info_once(
        "Prefill shared-read-done fastpath active (%s)",
        type(model_runner.attn_backend).__name__,
    )
    read_done = device_module.Event()
    read_done.record()
    model_runner.shared_read_done_event = read_done
