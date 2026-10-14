"""Two-microbatch prefill for TP all-reduce models.

A large single-request extend is split into two halves that run layer by layer
in turn. Each half's tensor-parallel all-reduce runs on a side stream, on its own
NCCL communicator, while the other half computes. Opt in with
``SGLANG_PREFILL_MBO``; a model takes part by implementing
``supports_prefill_mbo()`` and the microbatch operations strategy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# KV pages, k-pool groups and KDA chunks of both halves must stay aligned.
_SPLIT_ALIGN = 256

_child_backends: Optional[List[AttentionBackend]] = None
_active = False
_microbatch = 0
_comm_stream: Optional[torch.cuda.Stream] = None
_comm_group = None
_comm_ranks: List[int] = []


def enabled(model_runner) -> bool:
    """Whether the target model of ``model_runner`` runs prefill microbatches."""
    if not envs.SGLANG_PREFILL_MBO.get() or model_runner.is_draft_worker:
        return False
    supports = getattr(model_runner.model, "supports_prefill_mbo", None)
    return supports is not None and supports()


def setup(child_backends: List[AttentionBackend], tp_ranks: List[int]) -> None:
    """Called once on every TP rank in the same order (creates a process group)."""
    global _child_backends, _comm_group, _comm_ranks
    _child_backends = child_backends
    _comm_ranks = list(tp_ranks)
    _comm_group = torch.distributed.new_group(ranks=_comm_ranks, backend="nccl")


def active_child_backends() -> Optional[List[AttentionBackend]]:
    return _child_backends if _active else None


def current_microbatch() -> int:
    return _microbatch


def set_microbatch(index: int) -> None:
    global _microbatch
    _microbatch = index


def end_forward() -> None:
    global _active, _microbatch
    _active = False
    _microbatch = 0


def _split_tokens(forward_batch: ForwardBatch) -> Optional[int]:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if _child_backends is None or forward_batch.forward_mode != ForwardMode.EXTEND:
        return None
    if forward_batch.batch_size != 1 or (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    ):
        return None
    num_tokens = forward_batch.extend_seq_lens_cpu[0]
    if num_tokens < envs.SGLANG_PREFILL_MBO_MIN_TOKENS.get():
        return None
    if forward_batch.input_ids.shape[0] != num_tokens:
        return None
    prefix = forward_batch.extend_prefix_lens_cpu[0]
    split = (prefix + num_tokens // 2) // _SPLIT_ALIGN * _SPLIT_ALIGN - prefix
    if split < _SPLIT_ALIGN or num_tokens - split < _SPLIT_ALIGN:
        return None
    return split


_TRACK_FIELDS = (
    "mamba_track_indices",
    "mamba_track_mask",
    "mamba_track_seqlens",
    "mamba_prefill_track_mask_cpu",
    "mamba_track_seqlens_cpu",
)
_PRE_FORWARD_FIELDS = (
    "mamba_cow_src_indices",
    "mamba_cow_dst_indices",
    "mamba_clear_indices",
)


def maybe_split(forward_batch: ForwardBatch) -> bool:
    """Split an eligible extend into two children with their own attention metadata."""
    global _active
    _active = False
    split = _split_tokens(forward_batch)
    if split is None:
        return False
    # The tracked state's owner is chosen from the CPU track metadata.
    if forward_batch.mamba_track_mask is not None and (
        forward_batch.mamba_prefill_track_mask_cpu is None
        or forward_batch.mamba_track_seqlens_cpu is None
    ):
        return False

    from sglang.srt.batch_overlap.two_batch_overlap import (
        TboForwardBatchPreparer,
        _update_device_and_sum_field_from_cpu_field,
    )
    from sglang.srt.model_executor.forward_batch_info import compute_position
    from sglang.srt.runtime_context import attention_backends

    num_tokens = forward_batch.extend_seq_lens_cpu[0]
    prefix = forward_batch.extend_prefix_lens_cpu[0]

    # Track fields belong to the child whose tokens reach the tracked length;
    # copy-on-write and clears already ran on the parent before the forward.
    track = {k: getattr(forward_batch, k) for k in _TRACK_FIELDS}
    pre = {k: getattr(forward_batch, k) for k in _PRE_FORWARD_FIELDS}
    for k in (*_TRACK_FIELDS, *_PRE_FORWARD_FIELDS):
        setattr(forward_batch, k, None)
    try:
        non_padded = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                tbo_split_token_index=split, num_token_non_padded=num_tokens
            )
        )
        children = [
            TboForwardBatchPreparer.filter_batch(
                forward_batch,
                start_token_index=start,
                end_token_index=end,
                start_seq_index=0,
                end_seq_index=1,
                out_num_token_non_padded=non_padded[i : i + 1],
                out_num_token_non_padded_cpu=end - start,
            )
            for i, (start, end) in enumerate(((0, split), (split, num_tokens)))
        ]
    finally:
        for k, v in (*track.items(), *pre.items()):
            setattr(forward_batch, k, v)

    child_a, child_b = children
    child_a.extend_seq_lens_cpu = [split]
    child_b.extend_seq_lens_cpu = [num_tokens - split]
    for child in children:
        _update_device_and_sum_field_from_cpu_field(
            child, "extend_seq_lens_cpu", "extend_seq_lens", "extend_num_tokens"
        )
    child_a.seq_lens_cpu = child_a.seq_lens_cpu.clone()
    child_a.seq_lens_cpu[0] = prefix + split
    _update_device_and_sum_field_from_cpu_field(
        child_a, "seq_lens_cpu", "seq_lens", "seq_lens_sum"
    )
    child_b.extend_prefix_lens_cpu = [prefix + split]
    _update_device_and_sum_field_from_cpu_field(
        child_b, "extend_prefix_lens_cpu", "extend_prefix_lens", None
    )
    prefill_backend, _ = attention_backends()
    _, child_b.extend_start_loc = compute_position(
        prefill_backend,
        child_b.extend_prefix_lens,
        child_b.extend_seq_lens,
        child_b.extend_num_tokens,
    )

    tracks = track["mamba_track_mask"] is not None
    if tracks:
        tracked = track["mamba_prefill_track_mask_cpu"][0]
        owner = (
            child_a
            if not tracked or track["mamba_track_seqlens_cpu"][0] <= prefix + split
            else child_b
        )
        for k, v in track.items():
            setattr(owner, k, v)

    forward_batch.tbo_split_seq_index = 0
    forward_batch.tbo_children = children
    for backend, child in zip(_child_backends, children):
        backend.init_forward_metadata(child)
    _active = True
    return True


def start_all_reduce(tensor: torch.Tensor) -> torch.cuda.Event:
    """All-reduce ``tensor`` in place on the side stream; returns its completion event."""
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()
    _comm_stream.wait_stream(torch.cuda.current_stream())
    # Keep the block from being reused before the all-reduce ends, even if the
    # forward is abandoned before the sum is joined.
    tensor.record_stream(_comm_stream)
    with torch.cuda.stream(_comm_stream):
        torch.distributed.all_reduce(tensor, group=_comm_group)
        event = torch.cuda.Event()
        event.record()
    return event


class InFlightSum:
    """An all-reduce started on the side stream; complete() joins it."""

    __slots__ = ("value", "event")

    def __init__(self, value: torch.Tensor):
        self.value = value
        self.event = start_all_reduce(value)

    def complete(self) -> torch.Tensor:
        torch.cuda.current_stream().wait_event(self.event)
        return self.value


def start_pending_sum(stream) -> None:
    """Start the all-reduce a stage's recorded partial output owes."""
    from sglang.srt.layers.layer_boundary.layout import _sum_group
    from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum

    pending = stream.pending
    assert pending is not None and isinstance(pending.owed, DeclaredSum)
    assert list(_sum_group(pending.owed.group).ranks) == _comm_ranks
    pending.owed = InFlightSum(pending.value)
