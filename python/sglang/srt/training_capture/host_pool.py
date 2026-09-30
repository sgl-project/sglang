"""Bounded, reusable registered Host slots with explicit transfer ownership."""

from __future__ import annotations

import math
import threading
from typing import Protocol

import msgspec
import torch
from sglang.srt.training_capture.protocol import (
    DTYPES,
    ELEMENT_BYTES,
    CaptureError,
    KVSpec,
    aux_specs,
)


class BufferRegistrar(Protocol):
    def register(self, tensor: torch.Tensor) -> None: ...

    def unregister(self, tensor: torch.Tensor) -> None: ...


class HostSlot(msgspec.Struct, eq=False):
    storage: torch.Tensor
    tensors: dict[str, torch.Tensor]
    manifest_buffer: torch.Tensor
    state: str = "free"


class HostBufferPool:
    """Allocate/register once, then reuse only after both CUDA and Store completion.

    Quarantined slots keep strong references and their registrations until an
    operator closes the transport. A failed transfer never makes them reusable.
    """

    def __init__(
        self,
        *,
        kv: KVSpec,
        max_tokens: int,
        slots: int,
        max_bytes: int,
        registrar: BufferRegistrar,
        manifest_bytes: int = 1 << 20,
        pin_memory: bool = True,
    ):
        if min(max_tokens, slots, max_bytes, manifest_bytes) <= 0:
            raise ValueError("Host pool sizes must be positive")
        specs = aux_specs(max_tokens, max_tokens)
        for layer in kv.layers:
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            ):
                specs[f"target_{component}.{layer.layer_id}"] = (
                    kv.dtype,
                    [max_tokens, layer.num_kv_heads, dim],
                )
        layout = {}
        offset = 0
        for name, (dtype, shape) in specs.items():
            offset = (offset + 63) // 64 * 64
            length = math.prod(shape) * ELEMENT_BYTES[dtype]
            layout[name] = (offset, length, dtype, shape)
            offset += length
        manifest_offset = (offset + 63) // 64 * 64
        slot_bytes = manifest_offset + manifest_bytes
        if slot_bytes * slots > max_bytes:
            raise ValueError(
                f"Host pool requires {slot_bytes * slots} bytes, budget is {max_bytes}"
            )
        self.registrar = registrar
        self.lock = threading.Lock()
        self.slots: list[HostSlot] = []
        self.allocated_bytes = 0
        self.closed = False
        try:
            for _ in range(slots):
                storage = torch.empty(
                    slot_bytes, dtype=torch.uint8, pin_memory=pin_memory
                )
                views = {
                    name: storage[start : start + length]
                    .view(DTYPES[dtype])
                    .reshape(shape)
                    for name, (start, length, dtype, shape) in layout.items()
                }
                slot = HostSlot(storage, views, storage[manifest_offset:])
                self.registrar.register(storage)
                self.slots.append(slot)
                self.allocated_bytes += slot_bytes
        except Exception:
            self.close()
            raise

    def acquire(self) -> HostSlot | None:
        with self.lock:
            if self.closed:
                return None
            for slot in self.slots:
                if slot.state == "free":
                    slot.state = "filling"
                    return slot
        return None

    def release(self, slot: HostSlot, *, transfer_complete: bool) -> None:
        with self.lock:
            if not any(item is slot for item in self.slots) or slot.state != "filling":
                raise CaptureError("invalid or duplicate Host slot release")
            slot.state = "free" if transfer_complete else "quarantined"

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "allocated_bytes": self.allocated_bytes,
                **{
                    s: sum(slot.state == s for slot in self.slots)
                    for s in ("free", "filling", "quarantined")
                },
            }

    def close(self) -> None:
        with self.lock:
            if any(slot.state != "free" for slot in self.slots):
                raise CaptureError(
                    "cannot unregister buffers with outstanding or uncertain transfers"
                )
            # Keep failed registrations referenced if unregister itself fails.
            while self.slots:
                slot = self.slots[-1]
                self.registrar.unregister(slot.storage)
                self.slots.pop()
            self.allocated_bytes = 0
            self.closed = True
