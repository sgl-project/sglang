"""State the first forward otherwise creates lazily, built before graph capture
under CUDA graph persistence; no capture then records (or is broken by) its creation."""

from __future__ import annotations

import logging

import torch
from torch import nn

from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.foundry_adapter import get_foundry_adapter

logger = logging.getLogger(__name__)


def prepare_capture_state(
    *,
    model: nn.Module,
    dtype: torch.dtype,
    is_draft_worker: bool,
    is_ep_scale_joiner: bool,
) -> None:
    """Build the MoE dispatchers' buffers / expert maps and the logits all-gather state;
    without persistence the first warmup forward builds the same objects."""
    if not get_foundry_adapter().enabled:
        return
    # Lazy imports keep the runner package free of a layers import cycle.
    from sglang.srt.layers.logits_processor import prebuild_logits_gatherers
    from sglang.srt.layers.moe.token_dispatcher.base import prebuild_dispatchers

    # A joiner's EP buffers are collectives with ranks it meets only at the rendezvous;
    # they keep their lazy (or post-rendezvous) creation.
    num_dispatchers = 0 if is_ep_scale_joiner else prebuild_dispatchers(model)
    # A draft warms up after the target's graphs are captured,
    # past where persisted graphs need driver-placed symmetric memory; it stays lazy.
    num_gatherers = (
        0 if is_draft_worker else prebuild_logits_gatherers(model, dtype=dtype)
    )
    _probe_flashinfer_norm_cluster_size()
    if num_dispatchers or num_gatherers:
        log_info_on_rank0(
            logger,
            f"Prebuilt capture state: {num_dispatchers} MoE dispatchers, "
            f"{num_gatherers} logits gatherers.",
        )


def _probe_flashinfer_norm_cluster_size() -> None:
    # FlashInfer >= 0.7.4 loads a module at the first norm call, breaking SAVE capture;
    # no eager forward precedes it. Drop once FlashInfer probes outside capture.
    try:
        from flashinfer.norm.utils import get_sm_version, max_cluster_n
    except ImportError:
        return
    max_cluster_n(get_sm_version())
