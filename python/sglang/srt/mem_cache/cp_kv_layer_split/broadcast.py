"""Slot-keyed async broadcast plumbing for CP KV LayerSplit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from sglang.srt.layers.dp_attention import get_attention_cp_group

logger = logging.getLogger(__name__)


@dataclass
class PendingBroadcast:
    """State for one in-flight owner->peers broadcast."""

    stream: Optional[torch.cuda.Stream] = None
    work: object = None
    layer_id: Optional[int] = None


PYNCCL_PENDING_BROADCAST = object()


def get_pynccl_broadcast_comm():
    """Return the attention-CP PyNCCL communicator used for KV broadcasts."""
    group = get_attention_cp_group()
    pynccl_comm = getattr(group, "pynccl_comm", None)
    if pynccl_comm is None or not getattr(pynccl_comm, "available", False):
        raise RuntimeError(
            "CP KV LayerSplit requires an available PyNCCL communicator "
            "on the attention-CP group for KV broadcasts."
        )
    return pynccl_comm


def broadcast_inline(tensor: torch.Tensor, owner_cp: int, pynccl_comm) -> None:
    """Enqueue a broadcast on the current stream without side-stream state."""
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.broadcast(tensor, owner_cp)


class BroadcastSlots:
    """Per-pool slot registry for in-flight LayerSplit broadcasts."""

    def __init__(self, slot_kinds: Iterable[str], cp_rank: int) -> None:
        self._slots: dict[str, PendingBroadcast] = {
            kind: PendingBroadcast() for kind in slot_kinds
        }
        self._cp_rank = cp_rank

    def kinds(self) -> tuple[str, ...]:
        return tuple(self._slots.keys())

    def pending(self, kind: str) -> PendingBroadcast:
        try:
            return self._slots[kind]
        except KeyError:
            raise ValueError(f"unknown broadcast kind: {kind}")

    def start(
        self,
        kind: str,
        layer_id: int,
        tensor: torch.Tensor,
        owner_cp: int,
        pynccl_comm,
    ) -> None:
        """Launch an async broadcast on this slot's side stream."""
        self.clear(kind, next_layer_id=layer_id)
        ready_event = torch.cuda.current_stream().record_event()
        pending = self.pending(kind)
        if pending.stream is None:
            pending.stream = torch.cuda.Stream(device=tensor.device)
        pending.stream.wait_event(ready_event)
        with pynccl_comm.change_state(enable=True):
            pynccl_comm.broadcast(tensor, owner_cp, stream=pending.stream)
        pending.work = PYNCCL_PENDING_BROADCAST
        pending.layer_id = layer_id

    def clear(self, kind: str, next_layer_id: Optional[int] = None) -> None:
        """Wait on the slot's side stream so the current stream can proceed."""
        pending = self.pending(kind)
        if pending.work is None:
            return
        if next_layer_id is not None:
            logger.debug(
                "[cp-kv-layer-split] finishing pending %s broadcast before "
                "starting next one: cp_rank=%s pending_layer=%s next_layer=%s",
                kind,
                self._cp_rank,
                pending.layer_id,
                next_layer_id,
            )
        if pending.stream is not None:
            torch.cuda.current_stream().wait_stream(pending.stream)
        pending.work = None
        pending.layer_id = None

    def finish(self, kind: str, layer_id: int) -> None:
        """Wait for the pending broadcast; ``layer_id`` must match."""
        pending = self.pending(kind)
        if pending.work is None:
            return
        if pending.layer_id != layer_id:
            raise RuntimeError(
                "CP KV LayerSplit tried to finish an unexpected "
                f"{kind} broadcast: pending_layer={pending.layer_id}, "
                f"layer={layer_id}"
            )
        self.clear(kind)
