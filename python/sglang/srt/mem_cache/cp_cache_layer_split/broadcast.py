"""Slot-keyed async broadcast plumbing for CP Cache LayerSplit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch

from sglang.srt.distributed import get_attn_cp_group


@dataclass
class PendingBroadcast:
    """State for one in-flight owner->peers broadcast."""

    stream: Optional[torch.cuda.Stream] = None
    layer_id: Optional[int] = None
    active: bool = False


def get_pynccl_broadcast_comm():
    """Return the attention-CP PyNCCL communicator used for KV broadcasts."""
    group = get_attn_cp_group()
    pynccl_comm = getattr(group, "pynccl_comm", None)
    if pynccl_comm is None or not getattr(pynccl_comm, "available", False):
        raise RuntimeError(
            "CP Cache LayerSplit requires an available PyNCCL communicator "
            "on the attention-CP group for KV broadcasts."
        )
    return pynccl_comm


class BroadcastSlots:
    """Per-pool slot registry for in-flight LayerSplit broadcasts."""

    def __init__(self, slot_kinds: Iterable[str]) -> None:
        self._slots: dict[str, PendingBroadcast] = {
            kind: PendingBroadcast() for kind in slot_kinds
        }

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
        with self.launch(kind, layer_id, tensor.device):
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.broadcast(tensor, owner_cp)

    @contextmanager
    def launch(self, kind: str, layer_id: int, device: torch.device) -> Iterator[None]:
        """Run work on a slot's side stream and track it until consumption."""
        self.clear(kind)
        pending = self.pending(kind)
        device_module = torch.get_device_module(device)
        if pending.stream is None:
            pending.stream = device_module.Stream(device=device)
        pending.stream.wait_stream(device_module.current_stream(device))
        pending.layer_id = layer_id
        pending.active = True
        try:
            with device_module.stream(pending.stream):
                yield
        except Exception:
            self.clear(kind)
            raise

    def clear(self, kind: str) -> None:
        """Wait on the slot's side stream so the current stream can proceed."""
        pending = self.pending(kind)
        if not pending.active:
            return
        if pending.stream is not None:
            device_module = torch.get_device_module(pending.stream.device)
            device_module.current_stream(pending.stream.device).wait_stream(
                pending.stream
            )
        pending.layer_id = None
        pending.active = False

    def finish(self, kind: str, layer_id: int) -> None:
        """Wait for the pending broadcast; ``layer_id`` must match."""
        pending = self.pending(kind)
        if not pending.active:
            return
        if pending.layer_id != layer_id:
            raise RuntimeError(
                "CP Cache LayerSplit tried to finish an unexpected "
                f"{kind} broadcast: pending_layer={pending.layer_id}, "
                f"layer={layer_id}"
            )
        self.clear(kind)
